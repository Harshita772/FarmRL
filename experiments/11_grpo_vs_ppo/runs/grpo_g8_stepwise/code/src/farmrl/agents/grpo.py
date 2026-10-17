"""GRPO-style policy optimisation without a critic (own implementation).

The policy is the PPO policy of ``agents/ppo.py`` (same body, same action heads, same
observation normalisation, orthogonal init and storage-head mask) with the value head
removed. Instead of a learned baseline, the advantage comes from a *group* of episodes that
share a season: each update draws ``seasons_per_update`` training seasons and, for every
one, rolls out ``group_size`` episodes with the same weather and prices and different
pest-growth seeds, using the current policy.

``adv_mode``
    ``episode``   advantage = (episode return - group mean) / (group std + adv_eps), applied
                  to every step of the episode (the usual GRPO form);
    ``stepwise``  advantage at step t = (return-to-go from t - group mean at t) / (group std at
                  t + adv_eps), so each step is credited with what followed it. Episodes of
                  one group are aligned by step, so this is a per-step baseline taken from
                  the group instead of from a critic.

The update is PPO's: clipped probability ratio, an entropy bonus and, optionally,
``kl_coef`` times the estimated KL from the policy that collected the data (off by
default). The environment-step budget is counted in steps: an update uses
``seasons_per_update * group_size * T`` of them and training stops at ``total_timesteps``.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .ppo import ActorCritic

DEFAULTS = {"group_size": 8, "seasons_per_update": 0, "adv_mode": "episode", "adv_eps": 0.01, "kl_coef": 0.0}
TARGET_EPISODES = 168            # about 2048 steps per update, as in PPO, when seasons_per_update is 0


class PolicyNet(ActorCritic):
    """The PPO actor-critic network without its value head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.v

    def dist(self, obs):
        x = obs if self.norm is None else self.norm.normalize(obs)
        logits = torch.split(self.pi(self.body(x)), self.sizes, dim=-1)
        return [Categorical(logits=l, validate_args=False) for l in logits], None


def group_advantages(rewards, mode="episode", gamma=1.0, eps=0.01):
    """Advantages [G, T] for one group of G episodes (same season) from their rewards [G, T]."""
    r = np.asarray(rewards, dtype=float)
    togo = np.zeros_like(r)
    run = np.zeros(r.shape[0])
    for t in reversed(range(r.shape[1])):
        run = r[:, t] + gamma * run
        togo[:, t] = run
    if mode == "episode":
        ret = togo[:, 0]                                         # discounted return of the whole episode
        a = (ret - ret.mean()) / (ret.std() + eps)
        return np.repeat(a[:, None], r.shape[1], axis=1)
    if mode == "stepwise":
        return (togo - togo.mean(axis=0, keepdims=True)) / (togo.std(axis=0, keepdims=True) + eps)
    raise ValueError(f"unknown adv_mode '{mode}' (episode or stepwise)")


class GRPO:
    def __init__(self, env, cfg):
        self.env, self.c = env, {**DEFAULTS, **cfg}
        if int(self.c["group_size"]) < 2:
            raise ValueError("group_size must be at least 2")
        if not int(self.c["seasons_per_update"]):
            self.c["seasons_per_update"] = max(1, round(TARGET_EPISODES / int(self.c["group_size"])))
        nvec = [int(n) for n in env.action_space.nvec]
        masked = [len(nvec) - 1] if cfg.get("mask_storage") and getattr(env, "on", {}).get("storage") else []
        self.net = PolicyNet(env.observation_space.shape[0], nvec, cfg["hidden"],
                             obs_norm=cfg.get("obs_norm", False), ortho_init=cfg.get("ortho_init", False),
                             head=cfg.get("head", "factored"), masked_heads=masked,
                             last_stage=(env.T - 1.5) / env.T)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.log = []

    # ----------------------------------------------------------------- state / interface
    def state(self):
        return self.net.state_dict()

    def set_state(self, state):
        self.net.load_state_dict(state)

    def save(self, path): torch.save(self.state(), path)
    def load(self, path): self.set_state(torch.load(path, weights_only=True))

    def predict(self, obs, env=None):
        with torch.no_grad():
            ds, _ = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
            return self.net._decode([d.probs.argmax(-1) for d in ds]).numpy()

    def action_probs(self, obs):
        with torch.no_grad():
            ds, _ = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
            return [p.numpy() for p in self.net.marginals(ds)]

    # ----------------------------------------------------------------------- rollouts
    def episode(self, season, seed):
        """One episode of the current policy in ``season``: observations, actions, log-probs,
        rewards (environment units) and the episode's profit."""
        obs, _ = self.env.reset(seed=int(seed), options={"season": int(season)})
        O, A, L, R, done, info = [], [], [], [], False, {}
        while not done:
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, logp, _ = self.net.act(o)
            obs, r, term, trunc, info = self.env.step(a.numpy())
            O.append(o); A.append(a); L.append(logp); R.append(r)
            done = term or trunc
        return torch.stack(O), torch.stack(A), torch.stack(L), np.asarray(R, dtype=float), float(info["profit"])

    def collect(self, rng):
        """One update's data: ``seasons_per_update`` seasons x ``group_size`` episodes."""
        c = self.c
        seasons = self.env.seasons
        B, G = int(c["seasons_per_update"]), int(c["group_size"])
        chosen = rng.choice(seasons, size=B, replace=B > len(seasons))
        obs, act, logp, adv, returns, profits, stds = [], [], [], [], [], [], []
        for season in chosen:
            group = [self.episode(season, rng.integers(2 ** 31 - 1)) for _ in range(G)]
            rewards = np.stack([g[3] for g in group])                                  # [G, T]
            a = group_advantages(rewards, c["adv_mode"], c.get("gamma", 1.0), c["adv_eps"])
            for g, ai in zip(group, a):
                obs.append(g[0]); act.append(g[1]); logp.append(g[2]); adv.append(torch.as_tensor(ai, dtype=torch.float32))
            returns += list(rewards.sum(axis=1)); profits += [g[4] for g in group]
            stds.append(rewards.sum(axis=1).std())
        return {"obs": torch.cat(obs), "act": torch.cat(act), "logp": torch.cat(logp), "adv": torch.cat(adv),
                "returns": returns, "profits": profits, "group_std": float(np.mean(stds)), "seasons": [int(s) for s in chosen]}

    # ------------------------------------------------------------------------ update
    def update(self, batch):
        c = self.c
        obs, act, old_logp, adv = batch["obs"], batch["act"], batch["logp"], batch["adv"]
        stats = {"policy_loss": [], "entropy": [], "kl": []}
        for _ in range(int(c["epochs"])):
            for idx in torch.randperm(len(obs)).split(int(c["batch_size"])):
                logp, ent, _ = self.net.evaluate(obs[idx], act[idx])
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - c["clip_eps"], 1 + c["clip_eps"]) * adv[idx]
                kl = (old_logp[idx] - logp).mean()
                pi_loss = -torch.min(s1, s2).mean()
                loss = pi_loss - c.get("ent_coef", 0.01) * ent.mean() + float(c["kl_coef"]) * kl
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                for k, x in zip(stats, [pi_loss, ent.mean(), kl]):
                    stats[k].append(float(x))
        if self.net.norm is not None:                      # statistics move only between updates
            self.net.norm.update(obs)
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def learn(self, total_timesteps, log_fn=None, tensorboard_dir=None):
        """Train until ``total_timesteps`` environment steps are used. Returns the mean episode
        return per update; ``self.log`` has one dict per update (return, profit, entropy, KL,
        mean within-group std of the return and the wall-clock seconds since the start)."""
        c, T = self.c, self.env.T
        rng = np.random.default_rng(int(np.random.randint(2 ** 31 - 1)))
        per_update = int(c["seasons_per_update"]) * int(c["group_size"]) * T
        updates = max(1, int(total_timesteps) // per_update)
        history, self.log, start = [], [], time.time()
        for it in range(updates):
            if c.get("lr_decay"):
                for group in self.opt.param_groups:
                    group["lr"] = c["lr"] * (1.0 - it / updates)
            batch = self.collect(rng)
            stats = self.update(batch)
            history.append(float(np.mean(batch["returns"])))
            self.log.append({"iteration": it, "timesteps": (it + 1) * per_update, "mean_return": history[-1],
                             "mean_profit": float(np.mean(batch["profits"])), "policy_loss": stats["policy_loss"],
                             "value_loss": float("nan"), "entropy": stats["entropy"], "kl": stats["kl"],
                             "group_std": batch["group_std"], "wall_s": time.time() - start})
            if log_fn: log_fn(it, history[-1])
        return history
