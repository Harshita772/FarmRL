"""Yield response model: season totals of inputs -> wheat yield (quintal/ha).

    Y = y_max * [1 - exp(-n_coef (N + soil_n))]            Mitscherlich response to nitrogen
              * [1 - w_excess * max(0, (W - w_opt)/100)^2]   quadratic loss from excess water
              * [1 - ky * (1 - ET_a/ET_c)]                   FAO-33 loss from water deficit
              * [1 - pest_loss * D]                          loss from accumulated pest damage

N is nitrogen applied (kg/ha), W is rain + irrigation over the season (mm), ET_a/ET_c is
the season's actual / potential crop evapotranspiration and D in [0, 1] is pest damage.

``fit`` estimates y_max, n_coef, w_opt and w_excess from district-year records with
scipy's ``curve_fit``. A parameter the data cannot pin down (estimate on a bound, or a
standard error larger than the estimate) is NOT kept: it is reset to its prior below and
reported as ``identified: false``. ``ky`` is a literature constant and ``pest_loss`` is
only fitted when a pesticide table is supplied (``fit_pesticide``).
"""
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import curve_fit

# Priors used when the data cannot identify a parameter.
#   n_coef    0.012 /kg  -> ~83% of the plateau at 120 kg N/ha (the project's original placeholder)
#   w_opt     400 mm     -> about one season of wheat crop evapotranspiration at Ludhiana
#   pest_loss 0.35       -> the project's original placeholder for an unsprayed crop
#   ky        1.15       -> seasonal yield response factor for spring wheat, FAO Irrigation &
#                           Drainage Paper 33 (Doorenbos & Kassam 1979), also FAO-56 Table 24
PRIORS = {"y_max": 55.0, "n_coef": 0.012, "w_opt": 400.0, "w_excess": 0.0, "pest_loss": 0.35}
BOUNDS = {"y_max": (20.0, 90.0), "n_coef": (0.003, 0.06), "w_opt": (250.0, 600.0),
          "w_excess": (0.0, 1.0)}
FITTED = ["y_max", "n_coef", "w_opt", "w_excess"]


class YieldModel:
    def __init__(self, y_max=PRIORS["y_max"], n_coef=PRIORS["n_coef"], w_opt=PRIORS["w_opt"],
                 w_excess=PRIORS["w_excess"], pest_loss=PRIORS["pest_loss"], soil_n=30.0, ky=1.15,
                 y_scale=1.0):
        self.y_max = y_max          # fitted plateau of observed yields (q/ha)
        self.y_scale = y_scale      # env calibration factor, see env.calibrate_yield_scale
        self.n_coef = n_coef        # nitrogen response rate (1/kg)
        self.w_opt = w_opt          # season water supply above which yield falls (mm)
        self.w_excess = w_excess    # fractional loss per (100 mm of excess)^2
        self.pest_loss = pest_loss  # max fractional loss from pests
        self.soil_n = soil_n        # nitrogen supplied by the soil itself (kg/ha)
        self.ky = ky                # FAO yield response factor to water deficit
        self.report = None

    # ---------------------------------------------------------------- response curves
    def n_term(self, total_n):
        return 1 - np.exp(-self.n_coef * (np.asarray(total_n, dtype=float) + self.soil_n))

    def w_term(self, total_water):
        excess = np.maximum(0.0, (np.asarray(total_water, dtype=float) - self.w_opt) / 100.0)
        return np.clip(1 - self.w_excess * excess ** 2, 0.0, 1.0)

    def deficit_term(self, et_ratio):
        return np.clip(1 - self.ky * (1 - np.asarray(et_ratio, dtype=float)), 0.0, 1.0)

    def potential(self, total_n, total_water):
        """Yield with no water deficit and no pest damage."""
        return self.y_max * self.y_scale * self.n_term(total_n) * self.w_term(total_water)

    def final_yield(self, total_n, total_water, pest_damage, et_ratio=1.0):
        return float(self.potential(total_n, total_water) * self.deficit_term(et_ratio)
                     * (1 - self.pest_loss * pest_damage))

    # ----------------------------------------------------------------------- fitting
    def _curve(self, free, fixed):
        def f(X, *theta):
            p = {**fixed, **dict(zip(free, theta))}
            n, w = X
            excess = np.maximum(0.0, (w - p["w_opt"]) / 100.0)
            return (p["y_max"] * (1 - np.exp(-p["n_coef"] * (n + self.soil_n)))
                    * (1 - p["w_excess"] * excess ** 2))
        return f

    def _solve(self, X, y, free, fixed):
        p0 = [np.clip(getattr(self, k), *BOUNDS[k]) for k in free]
        lo, hi = zip(*(BOUNDS[k] for k in free))
        est, cov = curve_fit(self._curve(free, fixed), X, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        return dict(zip(free, est)), dict(zip(free, np.sqrt(np.diag(cov))))

    def fit(self, df, assumed_irrigation=300.0, n_col="n_kg_ha", rain_col="rain_mm",
            y_col="yield_q_ha", pest_df=None):
        """Estimate the response parameters from district-year records.

        ``df`` needs ``y_col`` (q/ha), ``n_col`` (kg N/ha) and ``rain_col`` (season rain,
        mm). The records carry no irrigation amounts, so every one is given the same
        ``assumed_irrigation`` (mm): W = rain + assumed_irrigation. Because those farms
        are irrigated, the deficit and pest terms are taken as 1 while fitting.
        Returns (and stores in ``self.report``) estimates, standard errors, which
        parameters were identified, R^2, RMSE, predictions and residuals.
        """
        d = df.dropna(subset=[y_col, n_col, rain_col])
        if len(d) < 8:
            raise ValueError(f"need at least 8 complete records to fit, got {len(d)}")
        n = d[n_col].to_numpy(float)
        w = d[rain_col].to_numpy(float) + assumed_irrigation
        y = d[y_col].to_numpy(float)

        self.y_scale = 1.0                   # the fit describes the records themselves
        est, se = self._solve((n, w), y, FITTED, {})
        identified = {}
        for k in FITTED:
            lo, hi = BOUNDS[k]
            on_bound = min(est[k] - lo, hi - est[k]) < 1e-3 * (hi - lo)
            identified[k] = bool(not on_bound and np.isfinite(se[k]) and se[k] < abs(est[k]))
        first_pass = {k: {"estimate": float(est[k]), "std_err": float(se[k])} for k in FITTED}

        # w_opt and w_excess describe one curve: without a usable slope there is no kink.
        if not identified["w_excess"] or not identified["w_opt"]:
            identified["w_excess"] = identified["w_opt"] = False
        fixed = {k: PRIORS[k] for k in FITTED if not identified[k]}
        free = [k for k in FITTED if identified[k]] or ["y_max"]
        fixed.pop("y_max", None)             # the plateau is always re-estimated
        if "y_max" not in free:
            free = ["y_max"] + free
        est, se = self._solve((n, w), y, free, fixed)
        for k in FITTED:
            setattr(self, k, float(est.get(k, fixed.get(k))))

        pred = self.potential(n, w)
        resid = y - pred
        ss_tot = float(((y - y.mean()) ** 2).sum())
        self.report = {
            "n_obs": int(len(y)),
            "r2": float(1 - (resid ** 2).sum() / ss_tot),
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "assumed_irrigation_mm": float(assumed_irrigation),
            "n_range_kg_ha": [float(n.min()), float(n.max())],
            "water_range_mm": [float(w.min()), float(w.max())],
            "mean_observed_yield": float(y.mean()),
            "identified": {k: bool(k in est) for k in FITTED},
            "std_err": {k: float(v) for k, v in se.items()},
            "first_pass": first_pass,
            "predicted": pred, "observed": y, "residuals": resid, "n": n, "water": w,
        }
        if pest_df is not None:
            self.fit_pesticide(pest_df)
        return self.report

    def fit_pesticide(self, pest_df, x_col="pesticide_index", y_col="yield_q_ha"):
        """Fit Y = a [1 - L exp(-k x)] to a (pesticide use, yield) table and adopt L as
        ``pest_loss`` if it is identified. Returns a small report either way."""
        d = pest_df.dropna(subset=[x_col, y_col])
        x, y = d[x_col].to_numpy(float), d[y_col].to_numpy(float)

        def f(x, a, loss, k):
            return a * (1 - loss * np.exp(-k * x))

        lo, hi = [0.5 * y.max(), 0.0, 0.05], [3.0 * y.max(), 0.9, 20.0]
        est, cov = curve_fit(f, x, y, p0=[y.max(), 0.3, 1.0], bounds=(lo, hi), maxfev=20000)
        se = np.sqrt(np.diag(cov))
        pred = f(x, *est)
        on_bound = min(est[1] - lo[1], hi[1] - est[1]) < 1e-3
        ok = bool(not on_bound and np.isfinite(se[1]) and se[1] < est[1])
        if ok:
            self.pest_loss = float(est[1])
        rep = {"n_obs": int(len(y)), "estimate": float(est[1]), "std_err": float(se[1]),
               "identified": ok, "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
               "curve": [float(v) for v in est], "x": x, "observed": y, "predicted": pred}
        if self.report is not None:
            self.report["pesticide"] = rep
        return rep

    # ------------------------------------------------------------------- persistence
    def params(self):
        return {k: float(getattr(self, k)) for k in
                ("y_max", "n_coef", "w_opt", "w_excess", "pest_loss", "soil_n", "ky", "y_scale")}

    def save(self, path, source=""):
        """Write parameters (+ fit diagnostics, without the per-record arrays) as YAML."""
        out = {"params": self.params(), "source": source}
        if self.report is not None:
            out["fit"] = {k: v for k, v in self.report.items() if not isinstance(v, np.ndarray)}
            if "pesticide" in out["fit"]:
                out["fit"]["pesticide"] = {k: v for k, v in out["fit"]["pesticide"].items()
                                           if not isinstance(v, np.ndarray)}
            pest_fitted = out["fit"].get("pesticide", {}).get("identified", False)
            out["fit"]["identified"]["pest_loss"] = bool(pest_fitted)
        with open(path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**yaml.safe_load(f)["params"])

    @classmethod
    def from_config(cls, path):
        """Fitted parameters if ``path`` exists, otherwise the priors."""
        return cls.load(path) if path and Path(path).exists() else cls()
