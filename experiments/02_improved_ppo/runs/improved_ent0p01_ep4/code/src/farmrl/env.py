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

Four optional mechanisms make over-use and timing costly. Each is off unless its config
block says ``enabled: true`` (so a config without the blocks gives the earlier simulator,
with the earlier observation and action spaces); parameters, sources and assumptions are
listed in ``MECHANISMS`` below and in configs/default.yaml:
  * cfg["groundwater"]     pumping cost proportional to the lift, which grows with the
                           season's own pumping and from year to year
  * cfg["waterlogging"]    water the topsoil cannot hold drains slowly; while it stands,
                           aeration stress lowers yield through stage-specific Ky factors
  * cfg["pest_resistance"] every spray leaves the survivors harder to kill
  * cfg["storage"]         a fourth action at harvest: sell now or store 1-2 months at an
                           uncertain price, paying storage, interest and grain losses
"""
import copy
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

# Defaults of the optional mechanisms. Sources were written from memory and could not be
# re-checked online when this was added: verify the page/table before quoting a number.
MECHANISMS = {
    # Energy to lift 1 mm over 1 ha (10 m3) by H metres = 10 * 9810 * H J = 0.02725 * H kWh
    # (physics), divided by the pump-set efficiency. The lift is the depth to the water table
    # in the season's harvest year plus the drawdown of the season's own pumping,
    # cum_irrigation / (1000 * specific_yield) metres.
    "groundwater": {
        "enabled": False,
        "depth_m": 25.0,               # ASSUMED depth to water, central Punjab, in depth_year
        "depth_year": 2024,
        "decline_m_per_year": 0.5,     # project brief; cf. Rodell et al. 2009 (Nature 460:999):
                                       # 4 cm/yr of water over NW India = 0.33 m/yr at Sy 0.12
        "min_depth_m": 3.0,
        "specific_yield": 0.12,        # ASSUMED, alluvial aquifer (norms range about 0.08-0.16)
        "pump_efficiency": 0.35,       # ASSUMED overall efficiency of a farm pump set
        "energy_price_per_kwh": 6.0,   # ASSUMED cost of supplying a unit of farm power (INR)
    },
    # Aeration stress as in FAO Irrigation & Drainage Paper 66 / AquaCrop (Steduto et al. 2012):
    # it starts at the anaerobiosis point, 5 vol% below saturation for wheat, and is complete
    # at saturation. With 15 vol% of drainable pores (loam: saturation 46, field capacity 31)
    # the top 0.3 m tolerates 30 mm above field capacity and is saturated at 45 mm.
    # A step's rain and irrigation first refill that layer (it holds top_layer_mm of
    # plant-available water and is taken to be as full as the whole root zone); what it
    # cannot hold stands in it and drains at drain_mm_per_day.
    # The yield loss follows FAO Irrigation & Drainage Paper 33 (Doorenbos & Kassam 1979),
    # 1 - Y/Ym = Ky (1 - ETa/ETm) per growth period, with ETa cut by the aeration stress and
    # the spring-wheat Ky of each period.
    "waterlogging": {
        "enabled": False,
        "drain_mm_per_day": 20.0,      # ASSUMED, set so that the standard calendar of four 75 mm
                                       # irrigations loses about 5% of its yield to waterlogging
        "top_layer_mm": 45.0,          # plant-available water of the top 0.3 m (15 vol%)
        "aeration_buffer_mm": 30.0,
        "saturation_mm": 45.0,
        "period_of_step": [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3],   # ASSUMED calendar, sown 1 Nov
        "ky_period": [0.2, 0.65, 0.55, 0.0],   # vegetative, flowering, yield formation, ripening
    },
    # Repeated use of one mode of action selects resistant survivors (IRAC / FRAC guidance;
    # in Punjab wheat: Phalaris minor and isoproturon, Malik & Singh 1995, Weed Technol. 9:419).
    # The size of the within-season effect is ASSUMED: each spray leaves the next one
    # (1 - efficacy_decay) times as effective.
    "pest_resistance": {"enabled": False, "efficacy_decay": 0.25},
    # Sale timing. The harvest sale pays the season price (MSP-like, certain). A stored crop
    # is sold at  P_h * exp(drift * m + x_m - x_0),  x the log deviation from trend, which
    # reverts by ``mean_reversion`` a month and receives a N(0, volatility) shock a month.
    # All six numbers are ASSUMED orders of magnitude: a lean-season rise of about 1% a
    # month, warehouse charges, grain losses below 1% of the crop in storage (ICAR-CIPHET,
    # Jha et al. 2015, put all storage channels together under 1%) and 7% a year on the
    # money tied up (the crop-loan rate).
    "storage": {
        "enabled": False,
        "max_months": 2,
        "monthly_drift": 0.01,
        "mean_reversion": 0.3,
        "monthly_volatility": 0.03,
        "cost_per_quintal_month": 10.0,
        "loss_per_month": 0.003,
        "interest_per_month": 0.0058,
    },
}
STEP_DAYS = 15

OBS_NAMES = ["stage", "soil_moisture", "soil_n", "pest", "cum_rain", "cum_n", "cum_water",
             "price_index", "last_rain", "last_et0", "et_ratio"]
# one extra observation per enabled mechanism that carries state through the season
EXTRA_OBS = {"groundwater": "pump_cost", "pest_resistance": "spray_efficacy"}


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
        self.mech = {k: {**v, **(cfg.get(k) or {})} for k, v in MECHANISMS.items()}
        self.on = {k: bool(v["enabled"]) for k, v in self.mech.items()}
        wl = self.mech["waterlogging"]
        if self.on["waterlogging"] and (len(wl["period_of_step"]) != self.T
                                        or max(wl["period_of_step"]) >= len(wl["ky_period"])):
            raise ValueError("waterlogging.period_of_step needs one index into ky_period per step")
        self._days = np.arange(STEP_DAYS) + 0.5
        gw = self.mech["groundwater"]
        self._pump_k = 0.02725 * gw["energy_price_per_kwh"] / gw["pump_efficiency"]   # INR per mm per m of lift
        self.store_levels = np.arange(int(self.mech["storage"]["max_months"]) + 1, dtype=float)
        self.obs_names = OBS_NAMES + [v for k, v in EXTRA_OBS.items() if self.on[k]]

        self.ym = yield_model or YieldModel.from_config(
            ROOT / cfg.get("fitted_params", "configs/fitted_params.yaml"))

        weather = self._load_weather(cfg) if weather is None else weather
        prices = self._load_prices(cfg) if prices is None else prices
        if weather.groupby("season").size().ne(self.T).any():
            raise ValueError(f"weather must have exactly {self.T} steps per season")
        self.weather = {int(s): g.sort_values("step")[["rain_mm", "et0_mm", "rh"]].to_numpy(float)
                        for s, g in weather.groupby("season")}
        self.dates = ({} if "start_date" not in weather else
                      {int(s): pd.to_datetime(g.sort_values("step")["start_date"]).tolist()
                       for s, g in weather.groupby("season")})
        self.rh_mean = float(weather["rh"].mean())
        self.heat = ({} if "heat_days" not in weather else
                     {int(s): float(v) for s, v in weather.groupby("season")["heat_days"].sum().items()})
        self.price_index = ({} if prices is None else
                            dict(zip(prices["season"].astype(int), prices["price_index"].astype(float))))

        train, test = dl.split_seasons(weather, cfg.get("train_end", dl.TRAIN_END))
        self.train_seasons, self.test_seasons = train, test
        self.split = split
        self.seasons = {"train": train, "test": test, "all": train + test}[split]
        if not self.seasons:
            raise ValueError(f"no seasons in split '{split}'")

        heads = [len(self.n_levels), len(self.w_levels), len(self.p_levels)]
        if self.on["storage"]:
            heads.append(len(self.store_levels))      # months to store, used at the last step only
        self.action_space = spaces.MultiDiscrete(heads)
        # see self.obs_names; every entry is scaled to roughly [0, 1]
        self.observation_space = spaces.Box(0.0, np.inf, shape=(len(self.obs_names),), dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_weather(cfg):
        site = cfg.get("weather_site", "ludhiana")
        path = dl.PROCESSED / f"weather_fortnightly_{site}.csv"
        if path.exists():
            return pd.read_csv(path)
        return dl.load_seasons(site, steps=cfg["steps_per_season"])   # needs data/raw

    @staticmethod
    def _load_prices(cfg):
        """The region's price table when it has been built, else the national one."""
        region = cfg.get("region")
        names = ([f"wheat_prices_{region.lower()}.csv"] if region else []) + ["wheat_prices.csv"]
        for name in names:
            if (dl.PROCESSED / name).exists():
                return pd.read_csv(dl.PROCESSED / name)
        try:
            return dl.load_prices(region=region)
        except FileNotFoundError:
            return None           # no price series on disk: every season pays base_price

    def step_dates(self, season=None):
        """Start date of every fortnight of ``season`` (default: the current one)."""
        season = self.season if season is None else int(season)
        if season in self.dates:
            return self.dates[season]
        start = pd.Timestamp(season, *dl.SEASON_START)
        return [start + pd.Timedelta(days=15 * t) for t in range(self.T)]

    def _obs(self):
        last_rain, last_et0 = (0.0, 0.0) if self.t == 0 else self._wx[self.t - 1, :2]
        et_ratio = 1.0 if self.cum_etc == 0 else self.cum_eta / self.cum_etc
        obs = [self.t / self.T, self.soil_moist, self.soil_n / 100, self.pest,
               self.cum_rain / 200, self.cum_n / 200, self.cum_water / 500,
               self.price / self.base_price, last_rain / 50, last_et0 / 100, et_ratio]
        if self.on["groundwater"]:
            obs.append(self._pump_k * self.pump_lift() / 20)      # INR per mm of the next irrigation
        if self.on["pest_resistance"]:
            obs.append(self.spray_eff)
        return np.array(obs, dtype=np.float32)

    def pump_lift(self):
        """Metres the next millimetre is lifted: the season's water table plus the drawdown
        of what has been pumped so far."""
        return self.depth + self.cum_irrigation / (1000.0 * self.mech["groundwater"]["specific_yield"])

    def _aeration_stress(self, excess):
        """Mean aeration stress (0-1) over a step that starts with ``excess`` mm standing in
        the topsoil and draining at the configured rate: 0 up to the buffer, 1 at saturation."""
        wl = self.mech["waterlogging"]
        standing = np.maximum(0.0, excess - wl["drain_mm_per_day"] * self._days)
        span = wl["saturation_mm"] - wl["aeration_buffer_mm"]
        return float(np.clip((standing - wl["aeration_buffer_mm"]) / span, 0.0, 1.0).mean())

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
        gw = self.mech["groundwater"]
        self.depth = max(gw["min_depth_m"],
                         gw["depth_m"] - gw["decline_m_per_year"] * (gw["depth_year"] - (self.season + 1)))
        self.cum_pump_cost = 0.0
        self.ponded = 0.0                                   # mm still standing in the topsoil
        self.last_stress = 0.0
        self.aeration = np.zeros(len(self.mech["waterlogging"]["ky_period"]))
        self.spray_eff = 1.0
        return self._obs(), {"season": self.season}

    def step(self, action):
        d = self.dyn
        n = self.n_levels[action[0]]
        w = self.w_levels[action[1]]
        p = self.p_levels[action[2]]
        rain, et0, rh = self._wx[self.t]

        # --- soil water bucket: inputs, drainage above capacity, then crop water use ---
        etc = d["kc"][self.t] * et0
        pump_cost = 0.0
        if self.on["groundwater"]:         # mean lift while this step's water is pumped
            lift = self.pump_lift() + 0.5 * w / (1000.0 * self.mech["groundwater"]["specific_yield"])
            pump_cost = self._pump_k * lift * w
        if self.on["waterlogging"]:        # what the topsoil cannot hold stands in it for a while
            wl = self.mech["waterlogging"]
            standing = max(0.0, self.ponded + rain + w - (1 - self.soil_moist) * wl["top_layer_mm"])
            self.last_stress = self._aeration_stress(standing)
            self.aeration[wl["period_of_step"][self.t]] += self.last_stress
            self.ponded = max(0.0, standing - wl["drain_mm_per_day"] * STEP_DAYS)
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
        kill = min(1.0, d["pest_kill_per_level"] * p * self.spray_eff)
        self.pest = float(np.clip(self.pest * growth * (1 - kill), 0, 1))
        if self.on["pest_resistance"] and p > 0:
            self.spray_eff *= 1 - self.mech["pest_resistance"]["efficacy_decay"]
        self.pest_damage = min(1.0, self.pest_damage + d["pest_damage_rate"] * self.pest / self.T)
        self.cum_pesticide += p

        cost = (n * self.costs["nitrogen_per_kg"] + w * self.costs["irrigation_per_mm"]
                + p * self.costs["pesticide_per_level"] + pump_cost)
        self.cum_pump_cost += pump_cost
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
            heat_days = self.heat.get(self.season, 0.0)
            y = self.ym.final_yield(self.cum_n_eff, self.cum_water, self.pest_damage, et_ratio, heat_days)
            waterlog_loss = 0.0
            if self.on["waterlogging"]:
                wl = self.mech["waterlogging"]
                steps = np.bincount(wl["period_of_step"], minlength=len(self.aeration))
                share = np.minimum(1.0, self.aeration / np.maximum(steps, 1))
                waterlog_loss = float(1 - np.prod(1 - np.asarray(wl["ky_period"]) * share))
                y *= 1 - waterlog_loss
            months, sale_price, storage_cost, sold = 0, self.price, 0.0, y
            if self.on["storage"]:
                st = self.mech["storage"]
                shocks = self.np_random.normal(size=len(self.store_levels) - 1)   # drawn whatever is chosen
                months = int(self.store_levels[action[3]]) if len(action) > 3 else 0
                x0 = x = float(np.log(self.price / self.base_price))
                for m in range(months):
                    x = (1 - st["mean_reversion"]) * x + st["monthly_volatility"] * shocks[m]
                sale_price = self.price * float(np.exp(st["monthly_drift"] * months + x - x0))
                sold = y * (1 - st["loss_per_month"]) ** months
                storage_cost = months * y * (st["cost_per_quintal_month"] + st["interest_per_month"] * self.price)
            revenue = sold * sale_price - storage_cost          # what the crop brings in, net of storage
            fixed = self.costs.get("fixed_per_ha", 0.0)
            reward += revenue - fixed
            info = {"yield": y, "revenue": revenue, "season": self.season, "price": self.price,
                    "sale_price": sale_price, "store_months": months, "storage_cost": storage_cost,
                    "pumping_cost": self.cum_pump_cost, "waterlog_loss": waterlog_loss,
                    "profit": revenue - fixed - self.cum_cost,      # INR/ha, before penalties
                    "input_cost": self.cum_cost, "penalty": self.cum_penalty,
                    "n_total": self.cum_n, "irrigation_total": self.cum_irrigation,
                    "pesticide_total": self.cum_pesticide, "rain_total": self.cum_rain,
                    "et_ratio": et_ratio, "pest_damage": self.pest_damage, "heat_days": heat_days}
        return self._obs(), reward / self.reward_scale, terminated, False, info


def scale_config(cfg, price=1.0, nitrogen_cost=1.0, irrigation_cost=1.0, pesticide_cost=1.0,
                 penalty=1.0, weather_site=None, fixed_cost=1.0, scale=None):
    """Copy of an env config with prices, unit costs and penalty weights multiplied, for
    what-if runs (sensitivity analysis, the dashboard's sliders, other districts).
    ``irrigation_cost`` also scales the pumping energy price; ``scale`` multiplies any other
    numeric entry, given as {"block.key": factor}, e.g. {"storage.monthly_drift": 1.2}."""
    out = copy.deepcopy(cfg)
    out["base_price"] = cfg["base_price"] * price
    for key, k in (("nitrogen_per_kg", nitrogen_cost), ("irrigation_per_mm", irrigation_cost),
                   ("pesticide_per_level", pesticide_cost), ("fixed_per_ha", fixed_cost)):
        if key in cfg["costs"]:
            out["costs"][key] = cfg["costs"][key] * k
    paths = dict(scale or {})
    if irrigation_cost != 1.0 and (cfg.get("groundwater") or {}).get("enabled"):
        paths["groundwater.energy_price_per_kwh"] = irrigation_cost
    for path, k in paths.items():
        block, key = path.split(".")
        default = (DYNAMICS if block == "dynamics" else MECHANISMS.get(block, {})).get(key)
        value = (cfg.get(block) or {}).get(key, default)
        out.setdefault(block, {})
        out[block] = dict(out[block] or {})
        out[block][key] = [v * k for v in value] if isinstance(value, list) else value * k
    out["penalties"] = {name: v * penalty for name, v in cfg["penalties"].items()}
    if weather_site is not None:
        out["weather_site"] = weather_site
    return out


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


def calibrate_ky(cfg, model, policy, bounds=(0.2, 2.5)):
    """Set the water-deficit response ``ky`` so that going without irrigation costs, on
    average over the train seasons, what the panel says a rainfed crop loses.

    The panel fit gives the loss of an unirrigated crop as a function of season rain
    (``model.rainfed_loss``); the simulator expresses deficit through the share of crop
    water demand met. Playing ``policy`` with irrigation switched off links the two:
    ky = mean fitted loss / mean simulated (1 - ET_a/ET_c). Returns the new ky, or leaves
    the literature value in place when the panel did not identify a rainfed loss."""
    if model.k_dry <= 0:
        return model.ky
    env = FarmEnv(cfg, split="train", yield_model=model)
    deficit, loss = [], []
    for season in env.seasons:
        obs, _ = env.reset(seed=season, options={"season": season})
        done = False
        while not done:
            action = np.asarray(policy(obs, env)).copy()
            action[1] = 0
            obs, _, done, _, info = env.step(action)
        deficit.append(1 - info["et_ratio"])
        loss.append(float(model.rainfed_loss(info["rain_total"])))
    if np.mean(deficit) <= 0:
        return model.ky
    model.ky = float(np.clip(np.mean(loss) / np.mean(deficit), *bounds))
    return model.ky
