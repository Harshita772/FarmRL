"""DQN with a branching dueling head (own implementation).

One Q-value per level of every input, from a shared trunk and a shared state value:
``Q_d(s, a_d) = V(s) + A_d(s, a_d) - mean_a A_d(s, a)`` for each action dimension d, so the
number of outputs grows with the sum, not the product, of the level counts (the idea of
branching dueling Q-networks). The learning target is shared: a double-DQN bootstrap
averaged over the branches that are active, using one reward for the whole joint action.

The storage branch only matters at the last fortnight (the environment reads it there);
with ``mask_storage`` it is trained and bootstrapped from only on those transitions.
Exploration is epsilon-greedy independently in every branch.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = {"hidden": 128, "lr": 5e-4, "gamma": 1.0, "buffer_size": 200_000, "batch_size": 128,
            "learning_starts": 5_000, "train_freq": 4, "target_update": 1_000,
            "eps_start": 1.0, "eps_end": 0.05, "eps_decay_frac": 0.5, "grad_clip": 10.0,
            "mask_storage": True, "log_every": 2048}


class BranchingQNet(nn.Module):
    def __init__(self, obs_dim, nvec, hidden=128):
        super().__init__()
        self.nvec = [int(n) for n in nvec]
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.adv = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                                nn.Linear(hidden // 2, n)) for n in self.nvec])

    def forward(self, obs):
        h = self.trunk(obs)
        v = self.value(h)
        return [v + a - a.mean(-1, keepdim=True) for a in (head(h) for head in self.adv)]


class DQN:
    def __init__(self, env, cfg):
        self.env, self.c = env, {**DEFAULTS, **cfg}
        self.nvec = [int(n) for n in env.action_space.nvec]
        self.net = BranchingQNet(env.observation_space.shape[0], self.nvec, self.c["hidden"])
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.c["lr"])
        self.last_stage = (env.T - 1.5) / env.T
        self.storage_head = len(self.nvec) - 1 if (self.c["mask_storage"] and getattr(env, "on", {}).get("storage")) else None

    # ----------------------------------------------------------------- state / interface
    def state(self):
        return self.net.state_dict()

    def set_state(self, state):
        self.net.load_state_dict(state)
        self.target.load_state_dict(state)

    def save(self, path): torch.save(self.state(), path)
    def load(self, path): self.set_state(torch.load(path, weights_only=True))

    def predict(self, obs, env=None):
        with torch.no_grad():
            qs = self.net(torch.as_tensor(obs, dtype=torch.float32))
            return np.array([int(q.argmax(-1)) for q in qs])

    def action_probs(self, obs):
        """Greedy one-hot per head (a Q-learner has no action distribution)."""
        a = self.predict(obs)
        return [np.eye(n, dtype=np.float32)[i] for n, i in zip(self.nvec, a)]

    def _active(self, obs):
        """[batch, heads] 1 where the head matters in this state (storage: last step only)."""
        w = torch.ones(obs.shape[0], len(self.nvec))
        if self.storage_head is not None:
            w[:, self.storage_head] = (obs[:, 0] > self.last_stage).float()
        return w

    # ----------------------------------------------------------------------- training
    def _update(self, batch):
        obs, act, rew, nxt, done = batch
        w, w_next = self._active(obs), self._active(nxt)
        qs = self.net(obs)
        with torch.no_grad():
            online_next = self.net(nxt)
            target_next = self.target(nxt)
            picked = torch.stack([t.gather(1, o.argmax(1, keepdim=True)).squeeze(1)
                                  for o, t in zip(online_next, target_next)], 1)
            boot = (picked * w_next).sum(1) / w_next.sum(1).clamp(min=1.0)
            y = rew + self.c["gamma"] * (1.0 - done) * boot
        taken = torch.stack([q.gather(1, act[:, i:i + 1]).squeeze(1) for i, q in enumerate(qs)], 1)
        td = F.smooth_l1_loss(taken, y.unsqueeze(1).expand_as(taken), reduction="none")
        loss = ((td * w).sum(1) / w.sum(1).clamp(min=1.0)).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.c["grad_clip"])
        self.opt.step()
        return float(loss), float(taken.mean())

    def learn(self, total_timesteps, log_fn=None, tensorboard_dir=None):
        """Train and return the mean episode return per logging window; ``self.log`` holds
        one dict per window (same columns as PPO's where they exist)."""
        c, n = self.c, int(total_timesteps)
        size = min(int(c["buffer_size"]), n)
        d = self.env.observation_space.shape[0]
        B = {"obs": np.zeros((size, d), np.float32), "nxt": np.zeros((size, d), np.float32),
             "act": np.zeros((size, len(self.nvec)), np.int64), "rew": np.zeros(size, np.float32),
             "done": np.zeros(size, np.float32)}
        pos = filled = 0
        obs, _ = self.env.reset()
        history, self.log, window, ep_ret = [], [], [], 0.0
        losses, qvals = [], []
        decay = max(1.0, c["eps_decay_frac"] * n)
        for step in range(1, n + 1):
            eps = c["eps_start"] + (c["eps_end"] - c["eps_start"]) * min(1.0, step / decay)
            greedy = self.predict(obs)
            explore = np.random.random(len(self.nvec)) < eps
            action = np.where(explore, [np.random.randint(k) for k in self.nvec], greedy)
            nxt, r, term, trunc, _ = self.env.step(action)
            end = term or trunc
            for key, val in (("obs", obs), ("nxt", nxt), ("act", action), ("rew", r), ("done", float(term))):
                B[key][pos] = val
            pos, filled = (pos + 1) % size, min(filled + 1, size)
            ep_ret += r
            obs = nxt
            if end:
                window.append(ep_ret); ep_ret = 0.0
                obs, _ = self.env.reset()
            if step >= c["learning_starts"] and step % c["train_freq"] == 0:
                idx = np.random.randint(filled, size=int(c["batch_size"]))
                batch = (torch.as_tensor(B["obs"][idx]), torch.as_tensor(B["act"][idx]),
                         torch.as_tensor(B["rew"][idx]), torch.as_tensor(B["nxt"][idx]),
                         torch.as_tensor(B["done"][idx]))
                loss, q = self._update(batch)
                losses.append(loss); qvals.append(q)
            if step % c["target_update"] == 0:
                self.target.load_state_dict(self.net.state_dict())
            if step % int(c["log_every"]) == 0 and window:
                history.append(float(np.mean(window)))
                self.log.append({"iteration": step // int(c["log_every"]) - 1, "timesteps": step,
                                 "mean_return": history[-1], "policy_loss": float("nan"),
                                 "value_loss": float(np.mean(losses)) if losses else float("nan"),
                                 "entropy": float("nan"), "kl": float("nan"),
                                 "epsilon": eps, "mean_q": float(np.mean(qvals)) if qvals else float("nan")})
                if log_fn: log_fn(len(history) - 1, history[-1])
                window, losses, qvals = [], [], []
        return history
