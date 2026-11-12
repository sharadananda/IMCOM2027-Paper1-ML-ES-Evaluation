"""Stronger baselines for Paper 1. Usage: python run_baselines.py SCENARIO OUTDIR  (20 replicates, seeds 20260+r as in the main experiment).
Added models: ses (per-unit exponential smoothing, alpha by in-sample one-step SSE), ar1_unit (per-unit AR(1)), fe_linear (pooled linear + unit fixed effects),
rf_cv / gb_cv (hyper-parameters chosen by forward-chaining validation on the last 2 observed months, retuned every 3rd origin).
Writes OUTDIR/baselines_<SCENARIO>.json"""
import sys, json, itertools, importlib, warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")
SC, OUT = sys.argv[1], sys.argv[2]
sys.argv = ["x", "0", "1", ".", SC]
import run_experiment_prospective as P
U = P.U
MODELS = ["persistence", "linear", "ses", "ar1_unit", "fe_linear", "rf", "gb", "rf_cv", "gb_cv"]
def ses_forecast(E, t):
    out = np.empty(U); grid = np.linspace(0.1, 0.9, 9)
    for u in range(U):
        x = E[u, :t + 1]; best = (1e18, 0.5)
        for a in grid:
            l = x[0]; sse = 0.0
            for v in x[1:]:
                sse += (v - l) ** 2; l = a * v + (1 - a) * l
            if sse < best[0]: best = (sse, a)
        a = best[1]; l = x[0]
        for v in x[1:]: l = a * v + (1 - a) * l
        out[u] = l
    return out
def ar1_unit(E, t):
    out = np.empty(U)
    for u in range(U):
        x = E[u, :t + 1]; X = x[:-1]; y = x[1:]; vx = X.var()
        b = np.cov(X, y)[0, 1] / vx if vx > 0 else 0.0; b = np.clip(b, -1, 1); c = y.mean() - b * X.mean(); out[u] = c + b * x[-1]
    return out
def cv_pick(make, grid, E, C, W, cs, t):
    best = None
    for prm in grid:
        err = 0.0
        for q in (t - 1, t):                    # forward chaining: train up to q-1, validate on target q  (targets observed by t)
            X, y = P.train_set(E, C, W, cs, q - 1); m = make(prm).fit(X, y)
            err += ((E[:, q] - m.predict(P.feats(E, C, W, cs, q))) ** 2).mean()
        if best is None or err < best[0]: best = (err, prm)
    return best[1]
def rep(r):
    seed = P.BASE_SEED + r; rng = np.random.default_rng(seed); mu, c0, w0, drift, bc, bw = P.draw(rng)
    E, C, W = P.simulate(rng, mu, c0, w0, drift, bc, bw); cs = np.cumsum(E, 1)
    sse = {m: [0.0, 0.0, 0, 0] for m in MODELS}; rf = gb = rfc = gbc = None; chosen = []
    for t in P.ORIGINS:
        s = t + 1; y = E[:, s]; post = s >= P.DRIFT_ON
        Xtr, ytr = P.train_set(E, C, W, cs, t); Xte = P.feats(E, C, W, cs, s)
        lin = LinearRegression().fit(Xtr, ytr)
        D = np.tile(np.eye(U), (t - 1, 1)); fe = LinearRegression().fit(np.hstack([Xtr, D]), ytr)
        if t in P.RETRAIN_RF_GB or rf is None:
            rf = RandomForestRegressor(80, max_depth=5, random_state=seed, n_jobs=1).fit(Xtr, ytr)
            gb = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=seed).fit(Xtr, ytr)
            pr = cv_pick(lambda p: RandomForestRegressor(80, max_depth=p[0], min_samples_leaf=p[1], random_state=seed, n_jobs=1), list(itertools.product((3, 6), (1, 10))), E, C, W, cs, t)
            pg = cv_pick(lambda p: GradientBoostingRegressor(n_estimators=p[0], max_depth=p[1], learning_rate=p[2], random_state=seed), list(itertools.product((60, 150), (2, 3), (0.05,))), E, C, W, cs, t)
            rfc = RandomForestRegressor(80, max_depth=pr[0], min_samples_leaf=pr[1], random_state=seed, n_jobs=1).fit(Xtr, ytr)
            gbc = GradientBoostingRegressor(n_estimators=pg[0], max_depth=pg[1], learning_rate=pg[2], random_state=seed).fit(Xtr, ytr); chosen.append([list(pr), list(pg)])
        Pm = {"persistence": E[:, t], "linear": lin.predict(Xte), "ses": ses_forecast(E, t), "ar1_unit": ar1_unit(E, t),
              "fe_linear": fe.predict(np.hstack([Xte, np.eye(U)])), "rf": rf.predict(Xte), "gb": gb.predict(Xte), "rf_cv": rfc.predict(Xte), "gb_cv": gbc.predict(Xte)}
        for m, p in Pm.items():
            i = 1 if post else 0; sse[m][i] += float(((y - p) ** 2).sum()); sse[m][2 + i] += U
    return {"sse": sse, "chosen": chosen}
if __name__ == "__main__":
    # usage: python run_baselines.py SCENARIO OUTDIR  -> one file per replicate (resumable), replicates 0..19 (seeds 20260+r)
    import os
    r0, r1 = int(os.environ.get("R0", 0)), int(os.environ.get("R1", 20)); step = int(os.environ.get("STEP", 1)); off = int(os.environ.get("OFF", 0))
    for r in range(r0 + off, r1, step):
        fn = f"{OUT}/bl_{SC}_{r}.json"
        if os.path.exists(fn): continue
        json.dump(rep(r), open(fn, "w")); print(SC, r, flush=True)
