"""Yield response model: season totals of inputs -> wheat yield (quintal/ha).

    Y = y_max * [1 - exp(-n_coef (N + soil_n))]            Mitscherlich response to nitrogen
              * [1 - w_excess * max(0, (W - w_opt)/100)^2]   quadratic loss from excess water
              * [1 - ky * (1 - ET_a/ET_c)]                   FAO-33 loss from water deficit
              * [1 - pest_loss * D]                          loss from accumulated pest damage
              * [1 - heat_coef * H]                          loss per heat-stress day in grain filling

N is nitrogen applied (kg/ha), W is rain + irrigation over the season (mm), ET_a/ET_c is
the season's actual / potential crop evapotranspiration and D in [0, 1] is pest damage.

``fit_panel`` estimates the response from a pooled multi-state panel (state fixed effects
for y_max, nitrogen, irrigated share, season rain, heat-stress days) and reports R^2 on
held-out years; ``fit`` is the earlier single-region form (nitrogen and rain only), kept
for comparison. Both use scipy's ``curve_fit``. A parameter the data cannot pin down (estimate on a bound, or a
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
# Panel fit: bounds, and the neutral value a parameter falls back to when not identified.
PANEL_BOUNDS = {"n_coef": (0.003, 0.08), "k_dry": (0.0, 0.95), "r0": (20.0, 800.0),
                "w_excess": (0.0, 1.0), "r_opt": (0.0, 400.0), "heat_coef": (0.0, 0.02)}
NO_RAIN_EFFECT = 1.0e6      # r0 this large makes the rainfed loss independent of rain
PANEL_NEUTRAL = {"n_coef": PRIORS["n_coef"], "k_dry": 0.0, "r0": NO_RAIN_EFFECT, "w_excess": 0.0,
                 "r_opt": 100.0, "heat_coef": 0.0}
PANEL_START = {"n_coef": 0.02, "k_dry": 0.4, "r0": 150.0, "w_excess": 0.03, "r_opt": 80.0,
               "heat_coef": 0.003}
A_BOUNDS = (3.0, 95.0)
BOUNDS = {"y_max": (20.0, 90.0), "n_coef": (0.003, 0.06), "w_opt": (250.0, 600.0),
          "w_excess": (0.0, 1.0)}
FITTED = ["y_max", "n_coef", "w_opt", "w_excess"]


class YieldModel:
    def __init__(self, y_max=PRIORS["y_max"], n_coef=PRIORS["n_coef"], w_opt=PRIORS["w_opt"],
                 w_excess=PRIORS["w_excess"], pest_loss=PRIORS["pest_loss"], soil_n=30.0, ky=1.15,
                 y_scale=1.0, heat_coef=0.0, k_dry=0.0, r0=100.0):
        self.y_max = y_max          # fitted plateau of observed yields (q/ha)
        self.y_scale = y_scale      # env calibration factor, see env.calibrate_yield_scale
        self.n_coef = n_coef        # nitrogen response rate (1/kg)
        self.w_opt = w_opt          # season water supply above which yield falls (mm)
        self.w_excess = w_excess    # fractional loss per (100 mm of excess)^2
        self.pest_loss = pest_loss  # max fractional loss from pests
        self.soil_n = soil_n        # nitrogen supplied by the soil itself (kg/ha)
        self.ky = ky                # yield response factor to water deficit (FAO-33 or calibrated)
        self.heat_coef = heat_coef  # fractional loss per heat-stress day (panel fit)
        self.k_dry = k_dry          # loss of a fully rainfed crop at zero rain (panel fit)
        self.r0 = r0                # mm of season rain over which that loss decays (panel fit)
        self.state_effects = None   # {state: plateau yield} from the panel fit
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

    def heat_term(self, heat_days):
        return np.clip(1 - self.heat_coef * np.asarray(heat_days, dtype=float), 0.1, 1.0)

    def rainfed_loss(self, rain):
        """Fitted fractional loss of an unirrigated crop that received ``rain`` mm."""
        return self.k_dry * np.exp(-np.asarray(rain, dtype=float) / self.r0)

    def final_yield(self, total_n, total_water, pest_damage, et_ratio=1.0, heat_days=0.0):
        return float(self.potential(total_n, total_water) * self.deficit_term(et_ratio)
                     * (1 - self.pest_loss * pest_damage) * self.heat_term(heat_days))

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

    # ------------------------------------------------------------------ pooled panel fit
    def _panel_curve(self, free, fixed, n_states):
        def f(X, *theta):
            state, n, irr, rain, heat = X
            p = {**fixed, **dict(zip(free, theta[n_states:]))}
            a = np.asarray(theta[:n_states])[state.astype(int)]
            excess = np.maximum(0.0, (rain - p["r_opt"]) / 100.0)
            return (a * (1 - np.exp(-p["n_coef"] * (n + self.soil_n)))
                    * (1 - p["k_dry"] * (1 - irr) * np.exp(-rain / p["r0"]))
                    * (1 - p["w_excess"] * irr * excess ** 2)
                    * np.clip(1 - p["heat_coef"] * heat, 0.1, 1.0))
        return f

    def _panel_solve(self, d, states, free, fixed, cols):
        idx = d[cols["state"]].map({s: i for i, s in enumerate(states)}).to_numpy(float)
        X = (idx, d[cols["n"]].to_numpy(float), d[cols["irr"]].to_numpy(float),
             d[cols["rain"]].to_numpy(float), d[cols["heat"]].to_numpy(float))
        y = d[cols["y"]].to_numpy(float)
        a0 = [float(np.clip(y[idx == i].mean() * 1.1, *A_BOUNDS)) for i in range(len(states))]
        lo = [A_BOUNDS[0]] * len(states) + [PANEL_BOUNDS[k][0] for k in free]
        hi = [A_BOUNDS[1]] * len(states) + [PANEL_BOUNDS[k][1] for k in free]
        est, cov = curve_fit(self._panel_curve(free, fixed, len(states)), X, y,
                             p0=a0 + [PANEL_START[k] for k in free], bounds=(lo, hi), maxfev=40000)
        se = np.sqrt(np.diag(cov))
        k = len(states)
        return (dict(zip(states, est[:k])), dict(zip(free, est[k:])), dict(zip(free, se[k:])),
                lambda dd: self._panel_curve(free, fixed, k)(
                    (dd[cols["state"]].map({s: i for i, s in enumerate(states)}).to_numpy(float),
                     dd[cols["n"]].to_numpy(float), dd[cols["irr"]].to_numpy(float),
                     dd[cols["rain"]].to_numpy(float), dd[cols["heat"]].to_numpy(float)), *est))

    @staticmethod
    def _r2(y, pred):
        return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    def fit_panel(self, df, region=None, assumed_irrigation=300.0, holdout_from=None,
                  state_col="state", n_col="n_kg_ha", irr_col="irrigated_share", rain_col="rain_mm",
                  heat_col="heat_days", y_col="yield_q_ha", year_col="year", pest_df=None):
        """Fit the pooled response on a multi-state district-year panel.

            Y = A_state [1 - exp(-n_coef (N + soil_n))] [1 - k_dry (1 - irr) exp(-rain / r0)]
                        [1 - w_excess irr max(0, (rain - r_opt)/100)^2] [1 - heat_coef * heat_days]

        ``A_state`` is a fixed effect per state (its plateau yield). Years from
        ``holdout_from`` on are first held out: the model is fitted on the earlier years
        and scored on them (R^2, RMSE, and the share of the variance *within states* that
        it explains beyond each state's training mean). It is then refitted on all years
        for use. The excess-rain loss is scaled by the irrigated share (rain on top of
        irrigation is what waterlogs a field). A parameter on a bound or with a standard
        error above its estimate is reset to its neutral value and the rest refitted, until
        every remaining one is identified. ``region`` selects which state's
        plateau becomes ``y_max`` (default: the area-unweighted mean of all states).
        """
        cols = {"state": state_col, "n": n_col, "irr": irr_col, "rain": rain_col, "heat": heat_col, "y": y_col}
        d = df.dropna(subset=list(cols.values())).copy()
        counts = d[state_col].value_counts()
        d = d[d[state_col].isin(counts[counts >= 8].index)]       # a fixed effect needs some rows
        states = sorted(d[state_col].unique())
        if len(d) < 40:
            raise ValueError(f"need at least 40 complete records for the panel fit, got {len(d)}")
        self.y_scale = 1.0
        free = list(PANEL_BOUNDS)

        def solve(data):
            keep, fixed, first, ok = list(free), {}, None, {}
            while True:
                a, est, se, predict = self._panel_solve(data, states, keep, fixed, cols)
                if first is None:
                    first = {k: {"estimate": float(est[k]), "std_err": float(se[k])} for k in keep}
                bad = []
                for k in keep:
                    lo, hi = PANEL_BOUNDS[k]
                    on_bound = min(est[k] - lo, hi - est[k]) < 1e-3 * (hi - lo)
                    if on_bound or not np.isfinite(se[k]) or se[k] >= abs(est[k]):
                        bad.append(k)
                if "k_dry" in bad and "r0" in keep:
                    bad.append("r0")                    # no rainfed loss -> nothing for rain to modulate
                if ("w_excess" in bad) != ("r_opt" in bad) and {"w_excess", "r_opt"} <= set(keep):
                    bad += ["w_excess", "r_opt"]        # threshold and slope describe one curve
                bad = sorted(set(bad))
                if not bad:
                    break
                fixed.update({k: PANEL_NEUTRAL[k] for k in bad})
                keep = [k for k in keep if k not in bad]
            ok = {k: k in keep for k in free}
            return a, {**fixed, **est}, se, ok, first, predict

        holdout = None
        if holdout_from is not None and (d[year_col] >= holdout_from).any() and (d[year_col] < holdout_from).sum() >= 40:
            train, test = d[d[year_col] < holdout_from], d[d[year_col] >= holdout_from]
            test = test[test[state_col].isin(train[state_col].unique())]
            _, _, _, _, _, predict = solve(train)
            y, pred = test[y_col].to_numpy(float), predict(test)
            state_mean = test[state_col].map(train.groupby(state_col)[y_col].mean()).to_numpy(float)
            holdout = {"from_year": int(holdout_from), "n_train": int(len(train)), "n_test": int(len(test)),
                       "r2_train": self._r2(train[y_col].to_numpy(float), predict(train)),
                       "r2_test": self._r2(y, pred), "rmse_test": float(np.sqrt(((y - pred) ** 2).mean())),
                       "r2_test_state_means_only": self._r2(y, state_mean),
                       "within_state_skill": float(1 - ((y - pred) ** 2).sum() / ((y - state_mean) ** 2).sum()),
                       "by_state": {s: self._r2(y[(test[state_col] == s).to_numpy()], pred[(test[state_col] == s).to_numpy()])
                                    for s in states if (test[state_col] == s).sum() >= 8}}
            holdout["_test"] = test.assign(predicted=pred)

        a, est, se, ok, first, predict = solve(d)
        self.state_effects = {s: float(v) for s, v in a.items()}
        match = [s for s in states if region and s.lower() == str(region).lower()]
        self.y_max = float(a[match[0]]) if match else float(np.mean(list(a.values())))
        self.n_coef, self.k_dry, self.r0 = float(est["n_coef"]), float(est["k_dry"]), float(est["r0"])
        self.w_excess, self.heat_coef = float(est["w_excess"]), float(est["heat_coef"])
        self.w_opt = float(est["r_opt"] + assumed_irrigation)
        y, pred = d[y_col].to_numpy(float), predict(d)
        test_rows = holdout.pop("_test") if holdout else None
        self.report = {
            "model": "pooled panel with state fixed effects",
            "n_obs": int(len(y)), "n_states": len(states), "n_districts": int(d["dist_code"].nunique()) if "dist_code" in d else None,
            "years": [int(d[year_col].min()), int(d[year_col].max())],
            "r2": self._r2(y, pred), "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
            "region": match[0] if match else None, "region_in_panel": bool(match),
            "assumed_irrigation_mm": float(assumed_irrigation),
            "mean_observed_yield": float(d.loc[d[state_col] == match[0], y_col].mean() if match else y.mean()),
            "n_range_kg_ha": [float(d[n_col].min()), float(d[n_col].max())],
            "identified": {**{k: bool(v) for k, v in ok.items()}, "y_max": True},
            "std_err": {k: float(v) for k, v in se.items()}, "first_pass": first,
            "r_opt_mm": float(est["r_opt"]), "state_effects": self.state_effects, "holdout": holdout,
            "predicted": pred, "observed": y, "residuals": y - pred, "data": d.assign(predicted=pred),
            "holdout_rows": test_rows,
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
                ("y_max", "n_coef", "w_opt", "w_excess", "pest_loss", "soil_n", "ky", "y_scale",
                 "heat_coef", "k_dry", "r0")}

    def save(self, path, source=""):
        """Write parameters (+ fit diagnostics, without the per-record arrays) as YAML."""
        out = {"params": self.params(), "source": source}
        if self.report is not None:
            out["fit"] = {k: v for k, v in self.report.items()
                          if not isinstance(v, np.ndarray) and not hasattr(v, "columns")}
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
