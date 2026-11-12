"""Aggregate replicate-level results; bootstrap CIs over replicates (B=2000, seed 7).
Usage: python aggregate_prospective.py [DIR]  ->  DIR/prospective_summary.json"""
import json, glob, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "/home/claude"
rng = np.random.default_rng(7); B = 2000
SC = ["linear", "nonlinear", "hetero", "regime", "combined"]
MODELS = ["persistence", "unit_mean", "roll3", "ar1", "linear", "rf", "gb", "static_linear"]
POL = ["lag_threshold", "roll3_threshold", "cycle_only", "forecast_linear", "forecast_gb", "forecast_static",
       "forecast_linear_AND_cycle", "forecast_gb_AND_cycle", "lag_AND_cycle"]


def ci(a):
    a = np.asarray(a, float); n = len(a)
    bs = [a[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pooled_ci(num_fn, recs):
    """bootstrap of a ratio statistic computed from pooled counts over resampled replicates"""
    n = len(recs); pt = num_fn(recs)
    bs = [num_fn([recs[i] for i in rng.integers(0, n, n)]) for _ in range(B)]
    return [pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def prf(recs, key, store="counts"):
    c = np.sum([r[store][key] for r in recs], axis=0); tp, fp, fn, tn = c
    p = tp / (tp + fp) if tp + fp else np.nan; rc = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * rc / (p + rc) if (p == p and rc == rc and p + rc) else np.nan
    return p, rc, f


def cost(c, w):
    return w * c[2] + c[1]


out = {}
for sc in SC:
    fs = sorted(glob.glob(f"{D}/prospective_raw_{sc}_*.json"))
    if not fs: continue
    recs = [r for f in fs for r in json.load(open(f))["recs"]]
    R = len(recs); S = {"R": R}
    S["n_cases"] = int(sum(r["n"] for r in recs))
    S["n_events"] = int(sum(r["counts"]["lag_threshold"][0] + r["counts"]["lag_threshold"][2] for r in recs))
    S["event_rate"] = ci([r["event_rate"] for r in recs])
    # ---- accuracy (rolling origin, whole period) + paired difference vs persistence
    arr = {m: np.array([np.sqrt((r["sse"][m][0] + r["sse"][m][1]) / (r["sse"][m][2] + r["sse"][m][3])) for r in recs]) for m in MODELS}
    acc = {}
    for m in MODELS:
        mae = [r["sse"][m][4] / (r["sse"][m][2] + r["sse"][m][3]) for r in recs]
        r2 = [1 - (r["sse"][m][0] + r["sse"][m][1]) / r["sst"] for r in recs]
        pre = [np.sqrt(r["sse"][m][0] / r["sse"][m][2]) for r in recs]
        post = [np.sqrt(r["sse"][m][1] / r["sse"][m][3]) for r in recs]
        acc[m] = {"rmse": ci(arr[m]), "mae": ci(mae), "r2": ci(r2), "rmse_pre": ci(pre), "rmse_post": ci(post),
                  "d_rmse_vs_persistence": ci(arr[m] - arr["persistence"]),
                  "d_rmse_vs_linear": ci(arr[m] - arr["linear"])}
    S["accuracy"] = acc
    # ---- group (unit) hold-out
    gm = ["persistence", "unit_mean", "linear", "rf", "gb"]
    garr = {m: np.array([np.sqrt(r["sse_group"][m][0] / r["sse_group"][m][1]) for r in recs]) for m in gm}
    S["group_holdout"] = {m: {"rmse": ci(garr[m]), "d_vs_persistence": ci(garr[m] - garr["persistence"])} for m in gm}
    # ---- alert policies
    pol = {}
    for k in POL:
        p, rc, f = prf(recs, k)
        pol[k] = {"precision": pooled_ci(lambda rs, k=k: prf(rs, k)[0], recs),
                  "recall": pooled_ci(lambda rs, k=k: prf(rs, k)[1], recs),
                  "f1": pooled_ci(lambda rs, k=k: prf(rs, k)[2], recs),
                  "n_alerts": int(sum(r["counts"][k][0] + r["counts"][k][1] for r in recs))}
        for w in (1, 3, 10):
            pol[k][f"cost_w{w}"] = ci([cost(r["counts"][k], w) for r in recs])
    S["policies"] = pol
    # pairwise value-add (cost per replicate, w=3), paired
    pw = {}
    for a, b in [("forecast_linear", "lag_threshold"), ("forecast_linear", "roll3_threshold"),
                 ("forecast_linear_AND_cycle", "forecast_linear"), ("forecast_linear_AND_cycle", "lag_AND_cycle"),
                 ("forecast_linear_AND_cycle", "lag_threshold"), ("forecast_gb", "forecast_linear"),
                 ("forecast_static", "forecast_linear")]:
        pw[f"{a} - {b}"] = {f"w{w}": ci([cost(r["counts"][a], w) - cost(r["counts"][b], w) for r in recs]) for w in (1, 3, 10)}
        pw[f"{a} - {b}"]["f1"] = ci([(lambda c: 2*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0)(r["counts"][a]) -
                                       (lambda c: 2*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0)(r["counts"][b]) for r in recs])
    S["pairwise"] = pw
    # post-drift phase
    ph = {}
    for k in ("lag_threshold", "forecast_linear", "forecast_static"):
        for phase in ("pre", "post"):
            ph[f"{k}|{phase}"] = {"f1": pooled_ci(lambda rs, k=k, phase=phase: prf(rs, f"{k}|{phase}", "counts_phase")[2], recs),
                                  "recall": pooled_ci(lambda rs, k=k, phase=phase: prf(rs, f"{k}|{phase}", "counts_phase")[1], recs)}
    S["phase"] = ph
    # ---- threshold sweep
    sw = {}
    for fam in ("lag", "forecast_linear", "forecast_gb", "forecast_linear_AND_cycle"):
        sw[fam] = {}
        for th in (64, 66, 68, 70, 72, 74, 76, 78):
            k = f"{fam}|{th}"
            sw[fam][th] = {"precision": prf(recs, k, "sweep")[0], "recall": prf(recs, k, "sweep")[1],
                           "f1": prf(recs, k, "sweep")[2],
                           "cost_w3": float(np.mean([cost(r["sweep"][k], 3) for r in recs])),
                           "cost_w10": float(np.mean([cost(r["sweep"][k], 10) for r in recs])),
                           "cost_w1": float(np.mean([cost(r["sweep"][k], 1) for r in recs]))}
    S["sweep"] = sw
    # ---- reviewer
    rv = {}
    base = {pk: {w: np.array([cost(r["counts"][pk], w) for r in recs]) for w in (1, 3, 10)} for pk in ("forecast_linear", "forecast_linear_AND_cycle")}
    for pk in ("forecast_linear", "forecast_linear_AND_cycle"):
        for st in (0.0, 0.5):
            for sg in (1, 2, 3, 4, 6, 8, 12):
                key = f"{pk}|{st}|{sg}"
                rv[key] = {f"d_cost_w{w}": ci([cost(r["review"][key], w) for r in recs] - base[pk][w]) for w in (1, 3, 10)}
                rv[key]["precision"] = prf(recs, key, "review")[0]; rv[key]["recall"] = prf(recs, key, "review")[1]
    S["reviewer"] = rv
    # ---- DEA
    dea = {}
    for w in recs[0]["dea"]:
        dea[w] = {k: ci([r["dea"][w][k] for r in recs]) for k in recs[0]["dea"][w]}
    S["dea"] = dea
    out[sc] = S
json.dump(out, open(f"{D}/prospective_summary.json", "w"), indent=1)
print("scenarios:", list(out))
