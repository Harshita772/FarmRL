"""Re-run the environment with a saved run's agent (used by the dashboard's simulator).

The simulation itself is ``rollout.run_episode`` on a ``FarmEnv`` built from the run's
config snapshot and yield parameters; this module only wires those together.
"""
import numpy as np
import pandas as pd

from . import rollout
from .agents.ppo import PPO
from .baselines import BASELINES, tuned_policies
from .env import FarmEnv, scale_config
from .yield_model import YieldModel


def make_env(run, split="all", **scales):
    """Environment as configured in the run, optionally with scaled prices/costs/penalties
    (keywords of ``env.scale_config``)."""
    cfg = scale_config(run.config["env"], **scales)
    saved = run.path / "prices.csv"                  # the price index the run was made with
    prices = pd.read_csv(saved) if saved.exists() else None
    return FarmEnv(cfg, prices=prices, split=split, yield_model=YieldModel.from_config(run.yield_params_path))


def load_agent(run):
    """The run's trained PPO agent."""
    agent = PPO(make_env(run), run.config["train"])
    agent.load(run.model_path)
    return agent


def policies(run, agent, names=None):
    """{name: policy function} for the trained agent, the registered baselines and the
    run's tuned rule baselines."""
    available = {"ppo": lambda obs, env: agent.predict(obs), **BASELINES, **tuned_policies(run.tuning)}
    return {k: v for k, v in available.items() if names is None or k in names}


def simulate(run, agent, season, names=None, seed=0, **scales):
    """Play ``season`` with each chosen policy; returns {policy: step table}."""
    env = make_env(run, **scales)
    return {name: rollout.run_episode(env, fn, season=season, seed=seed, policy_name=name)
            for name, fn in policies(run, agent, names).items()}


def action_distribution(run, agent, state):
    """Action probabilities for an observation given as {obs name: scaled value}; entries
    the run's environment has but ``state`` lacks take their start-of-season value.
    Returns {input: (levels in real units, probabilities)}."""
    env = make_env(run)
    start = dict(zip(env.obs_names, env.reset(seed=0)[0]))
    obs = np.array([state.get(k, start[k]) for k in env.obs_names], dtype=np.float32)
    probs = agent.action_probs(obs)
    levels = {"nitrogen_kg": env.n_levels, "irrigation_mm": env.w_levels, "pesticide_level": env.p_levels}
    if env.on["storage"]:
        levels["store_months"] = env.store_levels
    return {name: (lv, p) for (name, lv), p in zip(levels.items(), probs)}
