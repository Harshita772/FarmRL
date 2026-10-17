"""Shifted conditions for stress tests: a warmer climate, less rain, more volatile prices.

Each function returns a modified copy of the weather / price table (or env config) that is
handed to ``FarmEnv`` in place of the processed one, so the same trained policy can be
replayed under conditions it was not trained on. The physical assumptions are stated here
because they are assumptions: ET0 rises 4% per degree of warming (the order of the
sensitivity of temperature-based ET0 formulas), and the daily maximum temperature within a
fortnight is taken to be normal with a standard deviation of 3.5 degrees around the
fortnight's mean maximum, which turns a shift of the mean into extra heat-stress days
(days above 30 C between 15 February and 31 March, as in data_loader).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import data_loader as dl
from .env import scale_config

ET0_PER_DEGREE = 0.04
DAILY_SD = 3.5

# name -> (warming in C, rain multiplier, price-index deviation multiplier, storage volatility multiplier)
SCENARIOS = {
    "+1 °C": dict(warm=1.0),
    "+2 °C": dict(warm=2.0),
    "20% less rain": dict(rain=0.8),
    "2x price volatility": dict(price_vol=2.0, storage_vol=2.0),
    "+2 °C and 20% less rain": dict(warm=2.0, rain=0.8),
}


def _window_days(start, harvest_year):
    """Days of the fortnight starting at ``start`` that fall in the grain-filling heat window."""
    (m0, d0), (m1, d1) = dl.HEAT_WINDOW
    lo, hi = pd.Timestamp(harvest_year, m0, d0), pd.Timestamp(harvest_year, m1, d1)
    days = pd.date_range(start, periods=15)
    return int(((days >= lo) & (days <= hi)).sum())


def warm(weather, degrees):
    """Weather with every temperature raised by ``degrees``, ET0 scaled up and heat days recounted."""
    w = weather.copy()
    if not degrees:
        return w
    start = pd.to_datetime(w["start_date"])
    n = np.array([_window_days(t, s + 1) for t, s in zip(start, w["season"])])
    mu = w["t_max"].to_numpy(float)
    extra = n * (norm.cdf((mu + degrees - dl.HEAT_TMAX) / DAILY_SD) - norm.cdf((mu - dl.HEAT_TMAX) / DAILY_SD))
    for col in ("t_mean", "t_max", "t_min"):
        w[col] = w[col] + degrees
    w["et0_mm"] = w["et0_mm"] * (1.0 + ET0_PER_DEGREE * degrees)
    w["heat_days"] = w["heat_days"] + extra
    return w


def drier(weather, factor):
    w = weather.copy()
    w["rain_mm"] = w["rain_mm"] * factor
    return w


def volatile_prices(prices, factor):
    """Season price index with its deviation from 1 multiplied by ``factor`` (floored at 0.2)."""
    p = prices.copy()
    p["price_index"] = np.maximum(0.2, 1.0 + factor * (p["price_index"] - 1.0))
    return p


def apply(cfg_env, spec, weather=None, prices=None):
    """(cfg_env, weather, prices) with the scenario ``spec`` (a SCENARIOS value) applied.
    ``weather`` / ``prices`` default to the tables the config points to."""
    from .env import FarmEnv
    weather = FarmEnv._load_weather(cfg_env) if weather is None else weather
    prices = FarmEnv._load_prices(cfg_env) if prices is None else prices
    weather = warm(weather, spec.get("warm", 0.0))
    if spec.get("rain", 1.0) != 1.0:
        weather = drier(weather, spec["rain"])
    if spec.get("price_vol", 1.0) != 1.0 and prices is not None:
        prices = volatile_prices(prices, spec["price_vol"])
    if spec.get("storage_vol", 1.0) != 1.0 and (cfg_env.get("storage") or {}).get("enabled"):
        cfg_env = scale_config(cfg_env, scale={"storage.monthly_volatility": spec["storage_vol"]})
    return cfg_env, weather, prices
