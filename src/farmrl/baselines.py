"""Rule-based comparison policies. Each maps (obs, env) -> action indices.

An action is [nitrogen, irrigation, pesticide] plus, optionally, the months to store the
grain; that entry is read only at harvest by an environment with the storage mechanism
on, and leaving it out means selling at harvest.

``calendar_policy`` and ``threshold_policy`` build a rule from a parameter dict, so the
same two rule families can be played with their textbook settings (``recommended``,
``threshold``) or with settings tuned on the training seasons (farmrl.tuning).
"""
import numpy as np

# Fixed calendar, the same every year: 120 kg N/ha in three early splits, four full
# irrigations (crown-root initiation, tillering, booting, grain filling), one spray
# mid-season, grain sold at harvest.
CALENDAR_DEFAULT = {"n_steps": [0, 2, 4], "n_level": 2, "w_steps": [1, 3, 6, 8], "w_level": 2,
                    "p_steps": [6], "p_level": 1, "store": 0}
# Irrigate when dry, fertilise when soil N is low, spray when pests are high, sell at harvest.
THRESHOLD_DEFAULT = {"n_below": 40.0, "n_level": 1, "n_until": 12,
                     "w_full_below": 0.3, "w_half_below": 0.5, "w_until": 12,
                     "p_high_above": 0.3, "p_low_above": 0.15,
                     "store_below": 0.0, "store_months": 0}


def no_input(obs, env):
    return np.array([0, 0, 0])


def random_policy(obs, env):
    """Uniform random action, drawn from the env generator so runs are repeatable."""
    return env.np_random.integers(env.action_space.nvec)


def calendar_policy(params=None):
    """Calendar rule: doses at fixed fortnights whatever the season does.
    ``n_steps`` / ``w_steps`` / ``p_steps`` list the fortnights of each input, ``*_level``
    the dose index applied there, ``store`` the months the grain is kept after harvest."""
    c = {**CALENDAR_DEFAULT, **(params or {})}
    n_steps, w_steps, p_steps = set(c["n_steps"]), set(c["w_steps"]), set(c["p_steps"])

    def policy(obs, env):
        t = env.t
        action = [c["n_level"] if t in n_steps else 0, c["w_level"] if t in w_steps else 0,
                  c["p_level"] if t in p_steps else 0]
        return np.array(action + [c["store"]] if c["store"] else action)
    return policy


def threshold_policy(params=None):
    """Threshold rule: act on the state. Nitrogen while soil N is below ``n_below`` (until
    fortnight ``n_until``); a full / half irrigation below the two soil-moisture cut-offs
    (until ``w_until``); a high / low spray above the two pest cut-offs; store the grain
    for ``store_months`` when the harvest price index is below ``store_below``."""
    c = {**THRESHOLD_DEFAULT, **(params or {})}

    def policy(obs, env):
        t = env.t
        n = c["n_level"] if env.soil_n < c["n_below"] and t < c["n_until"] else 0
        w = 0
        if t < c["w_until"]:
            w = 2 if env.soil_moist < c["w_full_below"] else (1 if env.soil_moist < c["w_half_below"] else 0)
        p = 2 if env.pest > c["p_high_above"] else (1 if env.pest > c["p_low_above"] else 0)
        store = c["store_months"] if env.price / env.base_price < c["store_below"] else 0
        return np.array([n, w, p, store] if store else [n, w, p])
    return policy


recommended_schedule = calendar_policy()
recommended_schedule.__name__ = "recommended_schedule"
threshold_rule = threshold_policy()
threshold_rule.__name__ = "threshold_rule"

BASELINES = {"no_input": no_input, "random": random_policy,
             "recommended": recommended_schedule, "threshold": threshold_rule}
# rule families that can be tuned: name -> (policy factory, default parameters)
FAMILIES = {"recommended": (calendar_policy, CALENDAR_DEFAULT),
            "threshold": (threshold_policy, THRESHOLD_DEFAULT)}
TUNED_SUFFIX = "_tuned"


def tuned_policies(tuning):
    """{"<family>_tuned": policy} from a saved tuning record ({family: {"params": ...}})."""
    return {name + TUNED_SUFFIX: FAMILIES[name][0](rec["params"])
            for name, rec in (tuning or {}).items() if name in FAMILIES}
