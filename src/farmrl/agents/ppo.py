"""Minimal PPO for MultiDiscrete actions (own implementation)."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, nvec, hidden=64):
        super().__init__()
        self.nvec = list(nvec)
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, sum(self.nvec))
        self.v = nn.Linear(hidden, 1)

    def dist(self, obs):
        h = self.body(obs)
        logits = torch.split(self.pi(h), self.nvec, dim=-1)
        return [Categorical(logits=l, validate_args=False) for l in logits], self.v(h).squeeze(-1)

    def act(self, obs):
        ds, v = self.dist(obs)
        a = torch.stack([d.sample() for d in ds], -1)
        logp = sum(d.log_prob(a[..., i]) for i, d in enumerate(ds))
        return a, logp, v

    def evaluate(self, obs, a):
        ds, v = self.dist(obs)
        logp = sum(d.log_prob(a[..., i]) for i, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds)
        return logp, ent, v


class PPO:
    def __init__(self, env, cfg):
        self.env, self.c = env, cfg
        self.net = ActorCritic(env.observation_space.shape[0], env.action_space.nvec, cfg["hidden"])
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])

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

    def gae(self, buf, last_v):
        rew, done = np.array(buf["rew"]), np.array(buf["done"], dtype=float)
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
        for it in range(total_timesteps // self.c["n_steps"]):
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
        return [d.probs.numpy() for d in ds]

    def predict(self, obs):
        with torch.no_grad():
            ds, _ = self.net.dist(torch.as_tensor(obs, dtype=torch.float32))
        return np.array([d.probs.argmax().item() for d in ds])

    def save(self, path): torch.save(self.net.state_dict(), path)
    def load(self, path): self.net.load_state_dict(torch.load(path, weights_only=True))
