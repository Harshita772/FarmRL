"""Crop season MDP as a Gymnasium environment.

One episode is one wheat season at Ludhiana (sown 1 Nov), one step is a fortnight. Each
episode replays the real NASA POWER weather of one historical season and pays the real
price of that season (as an index around ``base_price``). ``split`` selects which seasons
can be drawn: "train" (sowing year <= ``train_end``), "test" (later years) or "all".

What is calibrated on data and what is assumed:
  * rain, reference ET, humidity per fortnight  -- NASA POWER, replayed as observed
  * price per season                            -- RBI/Agmarknet series via data_loader
  * yield response to N and to excess water     -- YieldModel, configs/fitted_params.yaml
  * costs                                       -- CACP cost of cultivation (configs/default.yaml)
  * soil water bucket, crop coefficients, N timing efficiency, pest growth, penalties
                                                -- textbook-style assumptions (cfg["dynamics"])
"""
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from . import data_loader as dl
from .yield_model import YieldModel

ROOT = Path(__file__).resolve().parents[2]

# Defaults for cfg["dynamics"]; every entry is an assumption, not a fitted value.
DYNAMICS = {
    # FAO-56 style crop coefficient per fortnight (initial 0.35 -> mid-season 1.15 -> maturity)
    "kc": [0.35, 0.40, 0.55, 0.75, 0.95, 1.10, 1.15, 1.15, 1.10, 0.85, 0.45, 0.20],
    "soil_capacity_mm": 150.0,     # plant-available water in the root zone
    "initial_moisture": 0.6,       # fraction of capacity after the pre-sowing irrigation
    "stress_fraction": 0.45,       # below this fraction of capacity ET is reduced (FAO-56, p = 0.55)
    # share of nitrogen applied in a fortnight that still reaches the grain
    "n_stage_efficiency": [1.0, 1.0, 1.0, 1.0, 0.9, 0.75, 0.5, 0.3, 0.15, 0.05, 0.0, 0.0],
    "n_carryover": 0.7,            # soil mineral N left after a fortnight of uptake
    "n_leach_fraction": 0.5,       # share of soil N lost when >= 50 mm drains in a fortnight
    "soil_n_initial": 30.0,
    "soil_n_safe": 80.0,           # kg/ha of mineral N above which the excess-N penalty applies
    "pest_initial": 0.05,
    "pest_growth": [0.1, 0.6],     # fortnightly growth ~ 1 + U(a, b) * (humidity / its mean)
    "pest_kill_per_level": 0.35,
    "pest_damage_rate": 0.5,
}

OBS_NAMES = ["stage", "soil_moisture", "soil_n", "pest", "cum_rain", "cum_n", "cum_water",
             "price_index", "last_rain", "last_et0", "et_ratio"]


class FarmEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict, weather=None, prices=None, split="train", yield_model=None):
        super().__init__()
        self.cfg = cfg
        self.T = cfg["steps_per_season"]
        self.n_levels = np.array(cfg["nitrogen_levels"], dtype=float)
        self.w_levels = np.array(cfg["irrigation_levels"], dtype=float)
        self.p_levels = np.array(cfg["pesticide_levels"], dtype=float)
        self.costs = cfg["costs"]
        self.pen = cfg["penalties"]
        self.base_price = cfg["base_price"]
        self.reward_scale = float(cfg.get("reward_scale", 1000.0))
        self.dyn = {**DYNAMICS, **cfg.get("dynamics", {})}
        if len(self.dyn["kc"]) != self.T or len(self.dyn["n_stage_efficiency"]) != self.T:
            raise ValueError("dynamics.kc and dynamics.n_stage_efficiency need one value per step")
        self.cap = float(self.dyn["soil_capacity_mm"])

        self.ym = yield_model or YieldModel.from_config(
            ROOT / cfg.get("fitted_params", "configs/fitted_params.yaml"))

        weather = self._load_weather(cfg) if weather is None else weather
        prices = self._load_prices() if prices is None else prices
        if weather.groupby("season").size().ne(self.T).any():
            raise ValueError(f"weather must have exactly {self.T} steps per season")
        self.weather = {int(s): g.sort_values("step")[["rain_mm", "et0_mm", "rh"]].to_numpy(float)
                        for s, g in weather.groupby("season")}
        self.rh_mean = float(weather["rh"].mean())
        self.price_index = ({} if prices is None else
                            dict(zip(prices["season"].astype(int), prices["price_index"].astype(float))))

        train, test = dl.split_seasons(weather, cfg.get("train_end", dl.TRAIN_END))
        self.train_seasons, self.test_seasons = train, test
        self.split = split
        self.seasons = {"train": train, "test": test, "all": train + test}[split]
        if not self.seasons:
            raise ValueError(f"no seasons in split '{split}'")

        self.action_space = spaces.MultiDiscrete(
            [len(self.n_levels), len(self.w_levels), len(self.p_levels)])
        # see OBS_NAMES; every entry is scaled to roughly [0, 1]
        self.observation_space = spaces.Box(0.0, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_weather(cfg):
        site = cfg.get("weather_site", "ludhiana")
        path = dl.PROCESSED / f"weather_fortnightly_{site}.csv"
        if path.exists():
            return pd.read_csv(path)
        return dl.load_seasons(site, steps=cfg["steps_per_season"])   # needs data/raw

    @staticmethod
    def _load_prices():
        path = dl.PROCESSED / "wheat_prices.csv"
        if path.exists():
            return pd.read_csv(path)
        try:
            return dl.load_prices()
        except FileNotFoundError:
            return None           # no price series on disk: every season pays base_price

    def _obs(self):
        last_rain, last_et0 = (0.0, 0.0) if self.t == 0 else self._wx[self.t - 1, :2]
        et_ratio = 1.0 if self.cum_etc == 0 else self.cum_eta / self.cum_etc
        return np.array([self.t / self.T, self.soil_moist, self.soil_n / 100, self.pest,
                         self.cum_rain / 200, self.cum_n / 200, self.cum_water / 500,
                         self.price / self.base_price, last_rain / 50, last_et0 / 100, et_ratio],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        """``options={"season": 2019}`` replays a chosen season; otherwise one is drawn
        uniformly from the split."""
        super().reset(seed=seed)
        season = (options or {}).get("season")
        if season is None:
            season = self.seasons[int(self.np_random.integers(len(self.seasons)))]
        self.season = int(season)
        self._wx = self.weather[self.season]
        self.t = 0
        self.soil_moist = float(self.dyn["initial_moisture"])
        self.soil_n = float(self.dyn["soil_n_initial"])
        self.pest = float(self.dyn["pest_initial"])
        self.pest_damage = 0.0
        self.cum_rain = self.cum_n = self.cum_n_eff = self.cum_water = self.cum_irrigation = 0.0
        self.cum_pesticide = self.cum_eta = self.cum_etc = self.cum_cost = self.cum_penalty = 0.0
        self.price = self.base_price * self.price_index.get(self.season, 1.0)
        return self._obs(), {"season": self.season}

    def step(self, action):
        d = self.dyn
        n = self.n_levels[action[0]]
        w = self.w_levels[action[1]]
        p = self.p_levels[action[2]]
        rain, et0, rh = self._wx[self.t]

        # --- soil water bucket: inputs, drainage above capacity, then crop water use ---
        etc = d["kc"][self.t] * et0
        avail = self.soil_moist * self.cap + rain + w
        drain = max(0.0, avail - self.cap - etc)
        avail -= drain
        eta = min(avail, etc * min(1.0, avail / (d["stress_fraction"] * self.cap)))
        self.soil_moist = float(min(self.cap, avail - eta) / self.cap)
        self.cum_eta += eta
        self.cum_etc += etc
        self.cum_rain += rain
        self.cum_irrigation += w
        self.cum_water += w + rain

        # --- nitrogen: late applications and applications washed out count for less ---
        leach = d["n_leach_fraction"] * min(1.0, drain / 50.0)
        self.cum_n += n
        self.cum_n_eff += n * d["n_stage_efficiency"][self.t] * (1 - leach)
        self.soil_n = max(0.0, d["n_carryover"] * self.soil_n * (1 - leach) + n)

        # --- pests grow faster in humid fortnights; spraying knocks them back ---
        growth = 1 + self.np_random.uniform(*d["pest_growth"]) * rh / self.rh_mean
        self.pest = float(np.clip(self.pest * growth * (1 - d["pest_kill_per_level"] * p), 0, 1))
        self.pest_damage = min(1.0, self.pest_damage + d["pest_damage_rate"] * self.pest / self.T)
        self.cum_pesticide += p

        cost = (n * self.costs["nitrogen_per_kg"] + w * self.costs["irrigation_per_mm"]
                + p * self.costs["pesticide_per_level"])
        penalty = (self.pen["excess_nitrogen"] * max(0.0, self.soil_n - d["soil_n_safe"])
                   + self.pen["pesticide_use"] * p + self.pen.get("irrigation_per_mm", 0.0) * w)
        self.cum_cost += cost
        self.cum_penalty += penalty
        reward = -(cost + penalty)

        self.t += 1
        terminated = self.t >= self.T
        info = {}
        if terminated:
            et_ratio = self.cum_eta / self.cum_etc
            y = self.ym.final_yield(self.cum_n_eff, self.cum_water, self.pest_damage, et_ratio)
            revenue = y * self.price
            fixed = self.costs.get("fixed_per_ha", 0.0)
            reward += revenue - fixed
            info = {"yield": y, "revenue": revenue, "season": self.season, "price": self.price,
                    "profit": revenue - fixed - self.cum_cost,      # INR/ha, before penalties
                    "input_cost": self.cum_cost, "penalty": self.cum_penalty,
                    "n_total": self.cum_n, "irrigation_total": self.cum_irrigation,
                    "pesticide_total": self.cum_pesticide, "rain_total": self.cum_rain,
                    "et_ratio": et_ratio, "pest_damage": self.pest_damage}
        return self._obs(), reward / self.reward_scale, terminated, False, info


def calibrate_yield_scale(cfg, model, target_yield, policy, replicates=20, seed0=0):
    """Scale factor that makes ``policy`` average ``target_yield`` over the train seasons.

    The fitted ``y_max`` is the plateau of yields farmers actually obtained, i.e. after
    the water stress and pest losses that normal practice still incurs. In the simulator
    those losses are applied explicitly, so the potential must sit above the plateau:
    ``y_scale`` is chosen so that standard practice reproduces the observed mean yield."""
    model.y_scale = 1.0
    env = FarmEnv(cfg, split="train", yield_model=model)
    yields = []
    for season in env.seasons:
        for k in range(replicates):
            obs, _ = env.reset(seed=seed0 + 1000 * season + k, options={"season": season})
            done = False
            while not done:
                obs, _, done, _, info = env.step(policy(obs, env))
            yields.append(info["yield"])
    model.y_scale = float(target_yield / np.mean(yields))
    return model.y_scale
