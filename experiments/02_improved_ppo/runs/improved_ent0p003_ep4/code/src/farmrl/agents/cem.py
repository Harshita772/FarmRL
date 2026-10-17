"""Cross-entropy method over the threshold rule's parameters (own implementation).

The agent is the threshold rule of ``baselines.threshold_policy``; training searches its ten
parameters with a diagonal Gaussian in a unit box that is refit to the best candidates each
iteration. Candidates are scored on the training seasons with the reward PPO is trained on,
so this is the same family as the grid-searched ``threshold_tuned`` baseline, found by a
different search with no hand-made grid.
"""
from types import SimpleNamespace

import numpy as np
import torch

from ..baselines import threshold_policy
from ..tuning import score

DEFAULTS = {"population": 64, "elite_frac": 0.2, "iterations": 40, "seeds": 2, "init_std": 0.35,
            "min_std": 0.03, "smoothing": 0.7}

# name, low, high, kind
SPACE = [("n_below", 0.0, 120.0, "float"), ("n_level", 0, 2, "int"), ("n_until", 0, 12, "int"),
         ("w_full_below", 0.0, 0.7, "float"), ("w_half_below", 0.0, 0.8, "float"), ("w_until", 6, 12, "int"),
         ("p_low_above", 0.0, 1.01, "float"), ("p_high_above", 0.0, 1.01, "float"),
         ("store_below", 0.9, 1.1, "float"), ("store_months", 0, 2, "int")]


def decode(u, ref_levels=None):
    """Unit vector in [0, 1]^10 -> threshold-rule parameter dict."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    p = {}
    for x, (name, lo, hi, kind) in zip(u, SPACE):
        v = lo + x * (hi - lo)
        p[name] = int(round(v)) if kind == "int" else float(v)
    p["w_full_below"], p["w_half_below"] = sorted((p["w_full_below"], p["w_half_below"]))
    p["p_low_above"], p["p_high_above"] = sorted((p["p_low_above"], p["p_high_above"]))
    return p


def encode(params):
    return np.array([(params[n] - lo) / (hi - lo) for n, lo, hi, _ in SPACE], dtype=float)


class ThresholdCEM:
    def __init__(self, env, cfg):
        self.env, self.c = env, {**DEFAULTS, **(cfg.get("cem") or {})}
        from ..baselines import THRESHOLD_DEFAULT
        self.mean = encode(THRESHOLD_DEFAULT)
        self.params = decode(self.mean)
        self.T = env.T
        self.log = []

    # ----------------------------------------------------------------- state / interface
    def state(self):
        return {"mean": torch.as_tensor(self.mean, dtype=torch.float64)}

    def set_state(self, state):
        self.mean = np.asarray(state["mean"], dtype=float)
        self.params = decode(self.mean)

    def save(self, path): torch.save(self.state(), path)
    def load(self, path): self.set_state(torch.load(path, weights_only=True))

    def _view(self, obs):
        """The state the rule reads, rebuilt from a scaled observation."""
        base = self.env.base_price
        return SimpleNamespace(t=int(round(float(obs[0]) * self.T)), soil_n=float(obs[2]) * 100.0,
                               soil_moist=float(obs[1]), pest=float(obs[3]), price=float(obs[7]) * base,
                               base_price=base, w_levels=self.env.w_levels, n_levels=self.env.n_levels,
                               p_levels=self.env.p_levels)

    def predict(self, obs, env=None):
        view = env if env is not None else self._view(obs)
        return np.asarray(threshold_policy(self.params)(obs, view)).astype(int)

    def action_probs(self, obs):
        a = self.predict(obs)
        nvec = [int(n) for n in self.env.action_space.nvec]
        a = np.concatenate([a, np.zeros(len(nvec) - len(a), dtype=int)])
        return [np.eye(n, dtype=np.float32)[i] for n, i in zip(nvec, a)]

    # ----------------------------------------------------------------------- training
    def _score(self, u, seed0):
        policy = threshold_policy(decode(u))
        return score(self.env, policy, self.env.seasons, self.c["seeds"], "reward", seed0=seed0)

    def learn(self, total_timesteps=None, log_fn=None, tensorboard_dir=None):
        """Search for ``iterations`` rounds (``total_timesteps`` is ignored: the budget is the
        cem block). Returns the best score per round in units of the environment's reward."""
        c, dim = self.c, len(SPACE)
        rng = np.random.default_rng(int(np.random.randint(2 ** 31 - 1)))
        mean, std = self.mean.copy(), np.full(dim, c["init_std"])
        n_elite = max(2, int(c["population"] * c["elite_frac"]))
        history = []
        best_u, best_s = mean.copy(), self._score(mean, 700_000)
        for it in range(int(c["iterations"])):
            pop = np.clip(rng.normal(mean, std, size=(int(c["population"]), dim)), 0.0, 1.0)
            pop[0] = best_u                                   # keep the best so far in the pool
            scores = np.array([self._score(u, 700_000 + 1_000_000 * (it + 1)) for u in pop])
            elite = pop[np.argsort(scores)[-n_elite:]]
            mean = c["smoothing"] * elite.mean(0) + (1 - c["smoothing"]) * mean
            std = np.maximum(c["smoothing"] * elite.std(0) + (1 - c["smoothing"]) * std, c["min_std"])
            if scores.max() > best_s:
                best_s, best_u = float(scores.max()), pop[int(scores.argmax())].copy()
            history.append(float(scores.max()) / self.env.reward_scale)
            self.log.append({"iteration": it, "timesteps": (it + 1) * int(c["population"]),
                             "mean_return": float(scores.mean()) / self.env.reward_scale,
                             "best_return": history[-1], "policy_loss": float("nan"), "value_loss": float("nan"),
                             "entropy": float(std.mean()), "kl": float("nan")})
            if log_fn: log_fn(it, history[-1])
        self.mean = best_u                                    # the best candidate seen, not the last mean
        self.params = decode(best_u)
        return history
