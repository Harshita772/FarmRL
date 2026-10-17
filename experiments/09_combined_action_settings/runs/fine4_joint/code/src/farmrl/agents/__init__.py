"""Learning agents. ``build_agent`` picks one from ``train.algo`` (default "ppo").

Every agent has ``learn``, ``predict(obs, env=None)``, ``action_probs(obs)``, ``state()`` /
``set_state()`` (a torch-saveable dict), ``save`` / ``load`` and a ``log`` list of dicts.
"""
ALGOS = ("ppo", "dqn", "cem_threshold", "grpo")


def build_agent(env, cfg_train):
    algo = (cfg_train or {}).get("algo", "ppo")
    if algo == "ppo":
        from .ppo import PPO
        return PPO(env, cfg_train)
    if algo == "dqn":
        from .dqn import DQN
        return DQN(env, cfg_train)
    if algo == "grpo":
        from .grpo import GRPO
        return GRPO(env, cfg_train)
    if algo == "cem_threshold":
        from .cem import ThresholdCEM
        return ThresholdCEM(env, cfg_train)
    raise ValueError(f"unknown train.algo '{algo}' (known: {ALGOS})")
