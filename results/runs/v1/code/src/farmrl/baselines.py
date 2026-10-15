"""Rule-based comparison policies. Each maps (obs, env) -> action indices."""
import numpy as np


def no_input(obs, env):
    return np.array([0, 0, 0])


def random_policy(obs, env):
    """Uniform random action, drawn from the env generator so runs are repeatable."""
    return env.np_random.integers(env.action_space.nvec)


def recommended_schedule(obs, env):
    """Fixed calendar, the same every year: 120 kg N/ha in three early splits, four full
    irrigations (crown-root initiation, tillering, booting, grain filling), one spray mid-season."""
    t = env.t
    n = 2 if t in (0, 2, 4) else 0
    w = 2 if t in (1, 3, 6, 8) else 0
    p = 1 if t == env.T // 2 else 0
    return np.array([n, w, p])


def threshold_rule(obs, env):
    """Irrigate when dry, fertilise when soil N low, spray when pests high."""
    n = 1 if env.soil_n < 40 else 0
    w = 2 if env.soil_moist < 0.3 else (1 if env.soil_moist < 0.5 else 0)
    p = 2 if env.pest > 0.3 else (1 if env.pest > 0.15 else 0)
    return np.array([n, w, p])


BASELINES = {"no_input": no_input, "random": random_policy,
             "recommended": recommended_schedule, "threshold": threshold_rule}
