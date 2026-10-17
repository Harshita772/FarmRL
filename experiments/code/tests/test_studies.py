"""Tests for the pieces added for the comparative studies: config overrides, the PPO options,
the joint head, DQN, CEM, the hindsight oracle and the grid-aware baselines."""
import copy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pytest
import torch

from farmrl import oracle, rollout
from farmrl.agents import build_agent
from farmrl.agents.cem import SPACE, decode, encode
from farmrl.agents.ppo import ActorCritic, PPO
from farmrl.baselines import BASELINES, THRESHOLD_DEFAULT, threshold_policy
from farmrl.utils import TRAIN_VARIANTS, apply_overrides, load_config
from tests.test_env import CFG, toy_env

TRAIN = load_config(ROOT / "configs/default.yaml")["train"]
SMALL = {**TRAIN, "n_steps": 48, "batch_size": 16, "epochs": 2, "hidden": 16}


def test_overrides_and_variant_bundle():
    cfg = {"train": {"gamma": 0.99, "ent_coef": 0.01, "variant": "improved"}}
    keys = apply_overrides(cfg, ["train.ent_coef=0.003", "train.gamma=0.97", "env.region=haryana"])
    assert cfg["train"]["ent_coef"] == 0.003 and cfg["env"]["region"] == "haryana"
    assert cfg["train"]["gamma"] == 0.97                       # explicit key beats the bundle
    assert cfg["train"]["total_timesteps"] == 1_000_000 and cfg["train"]["obs_norm"] is True
    assert set(keys) == {"train.ent_coef", "train.gamma", "env.region"}
    with pytest.raises(ValueError):
        apply_overrides({"train": {"variant": "nope"}})
    assert set(TRAIN_VARIANTS["improved"]) >= {"obs_norm", "ret_norm", "lr_decay", "ortho_init", "mask_storage"}


def test_default_ppo_keeps_the_original_parameter_names():
    net = PPO(toy_env(), TRAIN).net
    assert set(net.state_dict()) == {"body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias",
                                     "pi.weight", "pi.bias", "v.weight", "v.bias"}


def test_joint_head_encodes_and_marginalises():
    net = ActorCritic(5, [3, 3, 3, 3], hidden=8, head="joint_nw")
    assert net.sizes == [9, 3, 3]
    a = torch.tensor([[2, 1, 0, 2], [0, 2, 1, 0]])
    back = net._decode(net._encode(a))
    assert torch.equal(back, a)
    ds, _ = net.dist(torch.zeros(2, 5))
    marg = net.marginals(ds)
    assert [m.shape[-1] for m in marg] == [3, 3, 3, 3]
    assert all(torch.allclose(m.sum(-1), torch.ones(2)) for m in marg)
    joint = ds[0].probs.reshape(2, 3, 3)
    assert torch.allclose(marg[0], joint.sum(2)) and torch.allclose(marg[1], joint.sum(1))


def test_storage_head_only_counts_at_the_last_step():
    net = ActorCritic(4, [3, 3, 3, 3], hidden=8, masked_heads=[3], last_stage=(12 - 1.5) / 12)
    early, late = torch.tensor([[0.2, 0, 0, 0]]), torch.tensor([[11 / 12, 0, 0, 0]])
    assert [float(w) for w in net.weights(early)] == [1, 1, 1, 0]
    assert [float(w) for w in net.weights(late)] == [1, 1, 1, 1]
    a = torch.tensor([[1, 1, 1, 2]])
    lp_a, ent_a, _ = net.evaluate(early, a)
    lp_b, ent_b, _ = net.evaluate(early, torch.tensor([[1, 1, 1, 0]]))
    assert torch.allclose(lp_a, lp_b)                          # storage choice is invisible early on
    lp_c, _, _ = net.evaluate(late, a)
    assert not torch.allclose(lp_c, net.evaluate(late, torch.tensor([[1, 1, 1, 0]]))[0])


@pytest.mark.parametrize("extra", [{}, {"variant": "improved", "total_timesteps": 96, "mask_storage": True},
                                   {"head": "joint_nw"}, {"obs_norm": True, "ret_norm": True, "lr_decay": True,
                                                          "ortho_init": True, "gamma": 1.0}])
def test_ppo_options_train_and_act(extra):
    cfg = {**SMALL, **extra}
    if cfg.get("variant"):
        apply_overrides({"train": cfg})
    env = toy_env()
    agent = PPO(env, cfg)
    agent.learn(96)
    assert len(agent.log) >= 1 and np.isfinite(agent.log[-1]["value_loss"])
    obs, _ = env.reset(seed=0)
    action = agent.predict(obs)
    assert action.shape == (4,) and all(0 <= a < n for a, n in zip(action, env.action_space.nvec))
    assert [p.shape[-1] for p in agent.action_probs(obs)] == [3, 3, 3, 3]
    clone = PPO(env, cfg)
    clone.set_state(agent.state())
    assert np.array_equal(clone.predict(obs), action)


def test_dqn_and_cem_run_through_the_factory():
    env = toy_env()
    dqn = build_agent(env, {**SMALL, "algo": "dqn", "learning_starts": 40, "buffer_size": 500,
                            "log_every": 48, "target_update": 20})
    dqn.learn(144)
    obs, _ = env.reset(seed=1)
    assert dqn.predict(obs).shape == (4,) and len(dqn.log) >= 1
    other = build_agent(env, {**SMALL, "algo": "dqn"})
    other.set_state(dqn.state())
    assert np.array_equal(other.predict(obs), dqn.predict(obs))
    cem = build_agent(env, {**SMALL, "algo": "cem_threshold", "cem": {"population": 6, "iterations": 2, "seeds": 1}})
    cem.learn()
    assert set(cem.params) == set(THRESHOLD_DEFAULT) and len(cem.log) == 2
    state = cem.state()
    fresh = build_agent(env, {**SMALL, "algo": "cem_threshold"})
    fresh.set_state(state)
    assert fresh.params == cem.params
    assert cem.predict(obs, env).shape[0] in (3, 4)


def test_cem_parameter_space_round_trips():
    p = decode(encode(THRESHOLD_DEFAULT))
    assert set(p) == {n for n, *_ in SPACE}
    assert p["n_level"] == THRESHOLD_DEFAULT["n_level"] and p["w_half_below"] >= p["w_full_below"]
    assert p["p_high_above"] >= p["p_low_above"]


def test_cem_view_matches_the_environment_state():
    env = toy_env()
    cem = build_agent(env, {**SMALL, "algo": "cem_threshold"})
    obs, _ = env.reset(seed=3, options={"season": 2015})
    for _ in range(4):
        obs, *_ = env.step(np.array([1, 1, 0, 0]))
    assert np.array_equal(cem.predict(obs), cem.predict(obs, env))


def test_oracle_beats_simple_plans_and_replays():
    env = toy_env("test")
    season = env.seasons[0]
    cfg = {"population": 40, "iterations": 6, "search_seeds": 2, "polish_seeds": 4, "polish_rounds": 1}
    plan, value = oracle.solve_season(env, season, cfg, np.random.default_rng(0))
    assert plan.shape == (env.T, 4) and np.all(plan[:-1, 3] == 0)
    nothing = oracle.plan_reward(env, season, np.zeros_like(plan), 4)
    assert value > nothing + 1000
    policy = oracle.plan_policy({season: {"plan": plan.tolist(), "reward": value}})
    ep = rollout.episodes(rollout.run_many(env, {"oracle": policy}, seasons=[season], seeds=2))
    assert ep["nitrogen_kg"].iloc[0] == pytest.approx(sum(env.n_levels[plan[:, 0]]))


def test_baselines_map_doses_onto_a_finer_water_grid():
    cfg = copy.deepcopy(CFG)
    cfg["irrigation_levels"] = [0, 25, 50, 75]
    env = toy_env(cfg=cfg)
    env.reset(seed=0, options={"season": 2015})
    act = BASELINES["recommended"]
    seen = []
    for t in range(env.T):
        a = act(None, env)
        seen.append(env.w_levels[a[1]])
        env.step(a)
    assert set(seen) <= {0.0, 75.0} and 75.0 in seen          # the calendar waters in full doses
    env.reset(seed=0, options={"season": 2015})
    half = threshold_policy({**THRESHOLD_DEFAULT, "w_full_below": 0.0, "w_half_below": 2.0})
    a = half(None, env)
    assert env.w_levels[a[1]] == 50.0                           # 37.5 mm -> 50 mm (ties go up)
    standard = toy_env()
    standard.reset(seed=0, options={"season": 2015})
    assert standard.w_levels[half(None, standard)[1]] == 37.5
