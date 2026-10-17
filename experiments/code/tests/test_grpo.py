"""Tests for the critic-free GRPO agent."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pytest
import torch

from farmrl.agents import build_agent
from farmrl.agents.grpo import GRPO, PolicyNet, group_advantages
from farmrl.agents.ppo import PPO
from farmrl.utils import apply_overrides, load_config
from tests.test_env import toy_env

TRAIN = load_config(ROOT / "configs/default.yaml")["train"]
SMALL = {**TRAIN, "algo": "grpo", "n_steps": 48, "batch_size": 16, "epochs": 2, "hidden": 16,
         "group_size": 3, "seasons_per_update": 2}


def test_policy_has_the_ppo_body_and_heads_but_no_value_head():
    ppo = PPO(toy_env(), SMALL).net.state_dict()
    grpo = GRPO(toy_env(), SMALL).net.state_dict()
    assert not any(k.startswith("v.") for k in grpo) and any(k.startswith("v.") for k in ppo)
    assert set(grpo) == {k for k in ppo if not k.startswith("v.")}
    assert all(grpo[k].shape == ppo[k].shape for k in grpo)
    net = PolicyNet(5, [3, 3, 3, 3], hidden=8)
    ds, v = net.dist(torch.zeros(2, 5))
    assert v is None and len(ds) == 4
    a, logp, none = net.act(torch.zeros(5))
    assert a.shape == (4,) and none is None


def test_episode_advantage_is_group_normalised_and_constant_over_the_episode():
    rewards = np.array([[0.1, 0.2, 5.0], [0.1, 0.2, 7.0], [0.1, 0.2, 9.0], [0.1, 0.2, 11.0]])
    adv = group_advantages(rewards, "episode", eps=1e-8)
    assert adv.shape == rewards.shape
    assert np.allclose(adv.mean(axis=0), 0) and np.allclose(adv.std(axis=0), 1, atol=1e-6)
    assert np.allclose(adv, adv[:, :1])                       # the same value on every step of an episode
    assert list(np.argsort(adv[:, 0])) == [0, 1, 2, 3]        # ordered by return
    flat = group_advantages(np.ones((4, 3)), "episode", eps=0.01)
    assert np.allclose(flat, 0.0)                              # no spread, no signal (and no blow-up)


def test_stepwise_advantage_uses_return_to_go_normalised_at_each_step():
    rewards = np.array([[1.0, 0.0, 4.0], [1.0, 2.0, 4.0], [1.0, 4.0, 4.0]])
    adv = group_advantages(rewards, "stepwise", gamma=1.0, eps=1e-8)
    togo = np.array([[5.0, 4.0, 4.0], [7.0, 6.0, 4.0], [9.0, 8.0, 4.0]])
    assert np.allclose(adv[:, 0], (togo[:, 0] - 7) / togo[:, 0].std())
    assert np.allclose(adv.mean(axis=0), 0)
    assert np.allclose(adv[:, 2], 0.0, atol=1e-6)             # identical last reward: nothing to credit
    mixed = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 4.0], [2.0, 1.0, 0.0]])           # early and late rewards trade off
    undiscounted = group_advantages(mixed, "stepwise", gamma=1.0, eps=1e-8)
    discounted = group_advantages(mixed, "stepwise", gamma=0.5, eps=1e-8)
    assert undiscounted[2, 0] < undiscounted[0, 0] < undiscounted[1, 0]                 # row 2 is the worst undiscounted...
    assert discounted[2, 0] > discounted[0, 0] and discounted[2, 0] > discounted[1, 0]   # ...and the best once the future is discounted
    with pytest.raises(ValueError):
        group_advantages(rewards, "other")


def test_a_group_shares_one_season_and_differs_only_by_pest_seed():
    env = toy_env()
    agent = GRPO(env, SMALL)
    seen = []
    real = env.reset
    env.reset = lambda seed=None, options=None: (seen.append((options["season"], seed)), real(seed=seed, options=options))[1]
    batch = agent.collect(np.random.default_rng(0))
    assert len(seen) == 2 * 3 and len(batch["returns"]) == 6
    for g in range(2):
        group = seen[3 * g:3 * g + 3]
        assert len({s for s, _ in group}) == 1 and len({k for _, k in group}) == 3      # one season, three pest seeds
    assert batch["obs"].shape[0] == 6 * env.T and batch["adv"].shape[0] == 6 * env.T
    assert set(batch["seasons"]) <= set(env.seasons)


@pytest.mark.parametrize("extra", [{}, {"adv_mode": "stepwise"}, {"kl_coef": 0.1},
                                   {"variant": "improved", "group_size": 3, "seasons_per_update": 2, "epochs": 2,
                                    "batch_size": 16, "hidden": 16, "algo": "grpo"}])
def test_grpo_trains_logs_and_keeps_to_the_step_budget(extra):
    cfg = {**SMALL, **extra}
    if cfg.get("variant"):
        apply_overrides({"train": cfg}, [])
        cfg.update(group_size=3, seasons_per_update=2, epochs=2, batch_size=16, hidden=16)
    env = toy_env()
    agent = build_agent(env, cfg)
    assert isinstance(agent, GRPO)
    per_update = 2 * 3 * env.T
    hist = agent.learn(3 * per_update + 5)                    # three whole updates fit in the budget
    assert len(hist) == 3 and [r["timesteps"] for r in agent.log] == [per_update, 2 * per_update, 3 * per_update]
    row = agent.log[-1]
    assert np.isfinite(row["policy_loss"]) and row["entropy"] > 0 and row["wall_s"] > 0 and "mean_profit" in row
    assert np.isnan(row["value_loss"])                        # no critic
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs)
    assert action.shape == (4,) and [p.shape[-1] for p in agent.action_probs(obs)] == [3, 3, 3, 3]
    clone = GRPO(env, cfg)
    clone.set_state(agent.state())
    assert np.array_equal(clone.predict(obs), action)


def test_default_budget_matches_ppo_batch_size_and_bad_group_size_is_rejected():
    env = toy_env()
    g8, g16 = GRPO(env, {**SMALL, "group_size": 8, "seasons_per_update": 0}), GRPO(env, {**SMALL, "group_size": 16, "seasons_per_update": 0})
    assert g8.c["seasons_per_update"] == 21 and g16.c["seasons_per_update"] == 10
    assert 2000 <= g8.c["seasons_per_update"] * 8 * env.T <= 2100
    with pytest.raises(ValueError):
        GRPO(env, {**SMALL, "group_size": 1})
