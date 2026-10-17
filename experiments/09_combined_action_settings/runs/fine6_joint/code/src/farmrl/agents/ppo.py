"""Minimal PPO for MultiDiscrete actions (own implementation).

Every option below is off by default, which gives the original agent exactly:

    obs_norm      running mean / std of the observation, inside the network (so ``predict``
                  and saved weights carry it)
    ret_norm      rewards divided by the running std of the discounted return
    lr_decay      learning rate falls linearly to 0 over training
    ortho_init    orthogonal initial weights (gain sqrt 2 body, 0.01 policy, 1 value)
    mask_storage  the storage head only enters the loss at the last step of a season, the
                  only step where the environment reads it
    head          "factored" (one softmax per input) or "joint_nw" (one softmax over every
                  nitrogen x water pair; pesticide and storage keep their own)

``train.variant: improved`` switches on the first five and gamma = 1, 1M steps
(farmrl.utils.TRAIN_VARIANTS).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class RunningStat(nn.Module):
    """Running mean / variance of a vector, kept as buffers so it is saved with the weights."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x):
        x = x.reshape(-1, self.mean.numel()).float()
        n, m, v = x.shape[0], x.mean(0), x.var(0, unbiased=False)
        delta, total = m - self.mean, self.count + n
        self.mean += delta * n / total
        self.var.copy_((self.var * self.count + v * n + delta.pow(2) * self.count * n / total) / total)
        self.count.copy_(total)

    def normalize(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10.0, 10.0)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, nvec, hidden=64, obs_norm=False, ortho_init=False, head="factored",
                 masked_heads=(), last_stage=None):
        super().__init__()
        self.nvec = list(nvec)
        if head == "joint_nw":
            self.groups = [(0, 1)] + [(i,) for i in range(2, len(self.nvec))]
        elif head == "factored":
            self.groups = [(i,) for i in range(len(self.nvec))]
        else:
            raise ValueError(f"unknown head '{head}'")
        self.sizes = [int(np.prod([self.nvec[i] for i in g])) for g in self.groups]
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, sum(self.sizes))
        self.v = nn.Linear(hidden, 1)
        self.norm = RunningStat(obs_dim) if obs_norm else None
        # groups whose likelihood only counts at the last step of the season
        self.masked = [gi for gi, g in enumerate(self.groups) if all(h in set(masked_heads) for h in g)]
        self.last_stage = last_stage
        if ortho_init:
            for layer, gain in ((self.body[0], 2 ** 0.5), (self.body[2], 2 ** 0.5), (self.pi, 0.01), (self.v, 1.0)):
                nn.init.orthogonal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)

    def dist(self, obs):
        x = obs if self.norm is None else self.norm.normalize(obs)
        h = self.body(x)
        logits = torch.split(self.pi(h), self.sizes, dim=-1)
        return [Categorical(logits=l, validate_args=False) for l in logits], self.v(h).squeeze(-1)

    def weights(self, obs):
        """Per group weight of its log-probability / entropy: 1, or 1 only at the last step."""
        w = [torch.ones(obs.shape[:-1]) for _ in self.groups]
        if self.masked and self.last_stage is not None:
            last = (obs[..., 0] > self.last_stage).float()
            for gi in self.masked:
                w[gi] = last
        return w

    def _encode(self, a):
        """Environment action (one index per head) -> one index per group."""
        out = []
        for g in self.groups:
            idx = a[..., g[0]]
            for h in g[1:]:
                idx = idx * self.nvec[h] + a[..., h]
            out.append(idx)
        return out

    def _decode(self, idx):
        """One index per group -> environment action."""
        cols = [None] * len(self.nvec)
        for g, i in zip(self.groups, idx):
            for h in reversed(g[1:]):
                cols[h] = i % self.nvec[h]
                i = i // self.nvec[h]
            cols[g[0]] = i
        return torch.stack(cols, -1)

    def act(self, obs):
        ds, v = self.dist(obs)
        idx = [d.sample() for d in ds]
        w = self.weights(obs)
        logp = sum(d.log_prob(i) * wi for d, i, wi in zip(ds, idx, w))
        return self._decode(idx), logp, v

    def evaluate(self, obs, a):
        ds, v = self.dist(obs)
        idx, w = self._encode(a), self.weights(obs)
        logp = sum(d.log_prob(i) * wi for d, i, wi in zip(ds, idx, w))
        ent = sum(d.entropy() * wi for d, wi in zip(ds, w))
        return logp, ent, v

    def marginals(self, ds):
        """Probability of each level of each environment head (list of arrays)."""
        out = [None] * len(self.nvec)
        for g, d in zip(self.groups, ds):
            p = d.probs
            if len(g) == 1:
                out[g[0]] = p
            else:
                p = p.reshape(*p.shape[:-1], *[self.nvec[h] for h in g])
                for pos, h in enumerate(g):
                    axes = tuple(a - p.dim() for a in range(p.dim() - len(g), p.dim()) if a != p.dim() - len(g) + pos)
                    out[h] = p.sum(dim=axes)
        return out


class PPO:
    def __init__(self, env, cfg):
        self.env, self.c = env, cfg
        nvec = [int(n) for n in env.action_space.nvec]
        masked = [len(nvec) - 1] if cfg.get("mask_storage") and getattr(env, "on", {}).get("storage") else []
        self.net = ActorCritic(env.observation_space.shape[0], nvec, cfg["hidden"],
                               obs_norm=cfg.get("obs_norm", False), ortho_init=cfg.get("ortho_init", False),
                               head=cfg.get("head", "factored"), masked_heads=masked,
                               last_stage=(env.T - 1.5) / env.T)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.ret_stat = RunningStat(1) if cfg.get("ret_norm") else None
        self._disc_ret = 0.0

    # ----------------------------------------------------------------- state / interface
    def state(self):
        return self.net.state_dict()

    def set_state(self, state):
        self.net.load_state_dict(state)

    def collect(self, obs):
        buf = {k: [] for k in ["obs", "act", "logp", "rew", "done", "val"]}
        ep_returns, ep_ret = [], 0.0
        for _ in range(self.c["n_steps"]):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, logp, v = self.net.act(o)
            obs2, r, term, trunc, _ = self.env.step(a.numpy())
            for k, x in zip(buf, [o, a, logp, r, term or trunc, v]):
                buf[k].append(x)
            ep_ret += r
            obs = obs2
            if term or trunc:
                ep_returns.append(ep_ret); ep_ret = 0.0
                obs, _ = self.env.reset()
        with torch.no_grad():
            _, last_v = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
        return buf, obs, last_v, ep_returns

    def _scaled_rewards(self, rew, done):
        """Rewards divided by the running std of the discounted return (when ret_norm)."""
        if self.ret_stat is None:
            return rew
        run, series = self._disc_ret, np.empty_like(rew)
        for t in range(len(rew)):
            run = self.c["gamma"] * run + rew[t]
            series[t] = run
            if done[t]:
                run = 0.0
        self._disc_ret = run
        self.ret_stat.update(torch.as_tensor(series, dtype=torch.float32))
        return rew / float(torch.sqrt(self.ret_stat.var[0] + 1e-8))

    def gae(self, buf, last_v):
        rew, done = np.array(buf["rew"]), np.array(buf["done"], dtype=float)
        rew = self._scaled_rewards(rew, done)
        val = torch.stack(buf["val"]).numpy()
        adv, g = np.zeros_like(rew), 0.0
        for t in reversed(range(len(rew))):
            nv = last_v.item() if t == len(rew) - 1 else val[t + 1]
            delta = rew[t] + self.c["gamma"] * nv * (1 - done[t]) - val[t]
            g = delta + self.c["gamma"] * self.c["gae_lambda"] * (1 - done[t]) * g
            adv[t] = g
        return torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(adv + val, dtype=torch.float32)

    def update(self, buf, adv, ret):
        obs, act = torch.stack(buf["obs"]), torch.stack(buf["act"])
        old_logp = torch.stack(buf["logp"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}
        for _ in range(self.c["epochs"]):
            for idx in torch.randperm(n).split(self.c["batch_size"]):
                logp, ent, v = self.net.evaluate(obs[idx], act[idx])
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - self.c["clip_eps"], 1 + self.c["clip_eps"]) * adv[idx]
                pi_loss, v_loss = -torch.min(s1, s2).mean(), (ret[idx] - v).pow(2).mean()
                loss = (pi_loss + self.c.get("vf_coef", 0.5) * v_loss
                        - self.c.get("ent_coef", 0.01) * ent.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                for k, x in zip(stats, [pi_loss, v_loss, ent.mean(), (old_logp[idx] - logp).mean()]):
                    stats[k].append(x.item())
        if self.net.norm is not None:             # statistics move only between iterations, so the
            self.net.norm.update(obs)             # ratios inside one iteration see the same inputs
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def learn(self, total_timesteps, log_fn=None, tensorboard_dir=None):
        """Train and return the mean episode return per iteration. ``self.log`` keeps one
        dict per iteration (return, policy loss, value loss, entropy, approximate KL);
        ``tensorboard_dir`` additionally writes those scalars as TensorBoard events."""
        writer = None
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(tensorboard_dir))
        obs, _ = self.env.reset()
        history, self.log = [], []
        iterations = total_timesteps // self.c["n_steps"]
        for it in range(iterations):
            if self.c.get("lr_decay"):
                for group in self.opt.param_groups:
                    group["lr"] = self.c["lr"] * (1.0 - it / iterations)
            buf, obs, last_v, eps = self.collect(obs)
            adv, ret = self.gae(buf, last_v)
            stats = self.update(buf, adv, ret)
            if eps:
                history.append(float(np.mean(eps)))
                self.log.append({"iteration": it, "timesteps": (it + 1) * self.c["n_steps"],
                                 "mean_return": history[-1], **stats})
                if writer:
                    for k, v in self.log[-1].items():
                        writer.add_scalar(f"train/{k}", v, self.log[-1]["timesteps"])
                if log_fn: log_fn(it, history[-1])
        if writer:
            writer.close()
        return history

    def action_probs(self, obs):
        """Probability of each level of each input (list of arrays, one per action head)."""
        with torch.no_grad():
            ds, _ = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
            return [p.numpy() for p in self.net.marginals(ds)]

    def predict(self, obs, env=None):
        with torch.no_grad():
            ds, _ = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
            idx = [d.probs.argmax(-1) for d in ds]
            return self.net._decode(idx).numpy()

    def save(self, path): torch.save(self.state(), path)
    def load(self, path): self.set_state(torch.load(path, weights_only=True))
