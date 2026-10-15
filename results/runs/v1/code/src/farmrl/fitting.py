"""Fit and calibrate the yield model for a config (shared by scripts/prepare_data.py and
the notebook).

``fit_yield_model`` picks the pooled panel fit when the table supports it (irrigated share,
heat-stress days, more than one state) and the single-region form otherwise, scores the
earlier single-region form on the same held-out rows for comparison, then calibrates the
deficit response and the potential-yield scale for the config's region and weather site.
"""
import numpy as np

from .baselines import recommended_schedule
from .env import calibrate_ky, calibrate_yield_scale
from .yield_model import YieldModel


def compare_with_single_region(report, cut, irrigation, states):
    """Held-out-year scores of the single-region (nitrogen + rain) fit and of the pooled
    fit on the same rows, one entry per state in ``states`` with enough data."""
    out = {}
    if cut is None or report.get("holdout_rows") is None:
        return out
    for state in states:
        rows = report["data"][report["data"]["state"] == state]
        train, test = rows[rows["year"] < cut], rows[rows["year"] >= cut]
        if len(train) < 20 or len(test) < 8:
            continue
        old = YieldModel()
        old.fit(train, assumed_irrigation=irrigation)
        y = test["yield_q_ha"].to_numpy(float)
        old_pred = old.potential(test["n_kg_ha"], test["rain_mm"] + irrigation)
        held = report["holdout_rows"]
        new_pred = held[held["state"] == state]["predicted"].to_numpy(float)
        rmse = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
        out[state] = {"n_train": int(len(train)), "n_test": int(len(test)),
                      "single_region_r2_train": float(old.report["r2"]),
                      "single_region_r2_test": YieldModel._r2(y, old_pred),
                      "single_region_rmse_test": rmse(old_pred),
                      "pooled_r2_test": YieldModel._r2(y, new_pred), "pooled_rmse_test": rmse(new_pred)}
    return out


def fit_yield_model(cfg, panel, pest_df=None):
    """Returns (model, report). ``report["form"]`` is "pooled" or "single_region";
    ``report["comparison"]`` holds the single-region-vs-pooled scores (pooled form only)."""
    env_cfg, fit_cfg = cfg["env"], cfg["yield_fit"]
    irrigation, region = fit_cfg["assumed_irrigation_mm"], env_cfg.get("region")
    model = YieldModel()
    pooled = {"irrigated_share", "heat_days"} <= set(panel.columns) and panel["state"].nunique() > 1
    if pooled:
        report = model.fit_panel(panel, region=region, assumed_irrigation=irrigation,
                                 holdout_from=fit_cfg.get("holdout_from"), pest_df=pest_df)
        states = sorted({"Haryana", report["region"] or "Haryana"})
        report["comparison"] = compare_with_single_region(report, fit_cfg.get("holdout_from"), irrigation, states)
        report["ky_calibrated"] = bool(model.k_dry > 0)
        calibrate_ky(env_cfg, model, recommended_schedule)
    else:
        report = model.fit(panel, assumed_irrigation=irrigation, pest_df=pest_df)
        report["comparison"], report["ky_calibrated"] = {}, False
    report["form"] = "pooled" if pooled else "single_region"
    report["y_scale"] = calibrate_yield_scale(env_cfg, model, report["mean_observed_yield"], recommended_schedule)
    return model, report


def describe(model, report):
    """Lines of text summarising a fit (what prepare_data.py prints)."""
    lines = []
    if report["form"] == "pooled":
        h = report["holdout"]
        lines.append(f"yield model: pooled panel, {report['n_obs']} district-years, {report['n_states']} states "
                     "(state fixed effects)")
        lines.append(f"  in-sample R2 = {report['r2']:.3f}, RMSE = {report['rmse']:.2f} q/ha")
        if h:
            lines.append(f"  held-out years >= {h['from_year']}: R2 = {h['r2_test']:.3f}, RMSE = {h['rmse_test']:.2f} q/ha "
                         f"(n = {h['n_test']}, fitted on {h['n_train']})")
            lines.append(f"      state means alone give R2 = {h['r2_test_state_means_only']:.3f}; within states the "
                         f"model explains {h['within_state_skill']:+.1%} of the remaining variance")
        for state, c in report["comparison"].items():
            lines.append(f"  {state} rows only, held-out years (n = {c['n_test']}): single-region N+rain fit "
                         f"R2 = {c['single_region_r2_test']:+.3f} (RMSE {c['single_region_rmse_test']:.2f}); "
                         f"pooled fit R2 = {c['pooled_r2_test']:+.3f} (RMSE {c['pooled_rmse_test']:.2f})")
        if not report["region_in_panel"]:
            lines.append("  WARNING: the config's region is not a state of the panel; y_max is the mean of all states")
    else:
        lines.append(f"yield model: single-region form, R2 = {report['r2']:.3f}, RMSE = {report['rmse']:.2f} q/ha, "
                     f"n = {report['n_obs']} (no irrigated share / heat days / second state for a pooled fit)")
    names = {"w_opt": "r_opt"}
    for k, v in model.params().items():
        flag = report["identified"].get(names.get(k, k))
        note = "" if flag is None else ("  (fitted)" if flag else "  (not identified -> neutral/prior)")
        if k == "ky" and report["ky_calibrated"]:
            note = "  (calibrated to the fitted rainfed loss)"
        if k == "r0" and v > 1e5:
            v, note = float("inf"), "  (no rain dependence identified)"
        lines.append(f"  {k:<10} {v:10.4f}{note}")
    lines.append(f"  y_scale = {report['y_scale']:.3f}: the recommended schedule now averages the observed "
                 f"{report['mean_observed_yield']:.1f} q/ha ({report.get('region') or 'all records'}) on the train seasons")
    return lines
