"""H9: drift-fallback rule. Usage: python run_h9.py MODE OUTDIR   (MODE = tune | eval)
Rule: at origin t, compare trailing MSE (last K origins whose targets are already observed) of the retrained linear model and persistence;
if model MSE > (1+DELTA) * persistence MSE, issue persistence forecasts at this origin, else the model's. No look-ahead.
tune: seeds 900000+r, 10 replicates, grid over K,DELTA on regime+linear scenarios -> chosen K,DELTA written to OUTDIR/h9_params.json
eval: seeds 20260+r (same as main experiment), 20 replicates x 5 scenarios, chosen K,DELTA -> OUTDIR/h9_raw.json"""
import sys, json, itertools, importlib
import numpy as np
from sklearn.linear_model import LinearRegression
MODE, OUT = sys.argv[1], sys.argv[2]
THR = 70.0
def load(scen):
    sys.argv = ["x", "0", "1", ".", scen]
    import run_experiment_prospective as P; importlib.reload(P); return P
def rep(P, seed, K, D):
    rng = np.random.default_rng(seed); mu, c0, w0, drift, bc, bw = P.draw(rng); E, C, W = P.simulate(rng, mu, c0, w0, drift, bc, bw)
    cs = np.cumsum(E, 1); mse = {"lin": {}, "per": {}}
    out = {k: [0.0, 0.0, 0, 0] for k in ("lin", "per", "fb")}; cnt = {f"{k}|{ph}": [0, 0, 0, 0] for k in ("lin", "per", "fb") for ph in ("pre", "post")}
    fbn = {"pre": [0, 0], "post": [0, 0]}
    for t in P.ORIGINS:
        s = t + 1; y = E[:, s]; ev = y < THR; ph = "post" if s >= P.DRIFT_ON else "pre"
        Xtr, ytr = P.train_set(E, C, W, cs, t); lin = LinearRegression().fit(Xtr, ytr); pl = lin.predict(P.feats(E, C, W, cs, s)); pp = E[:, t]
        mse["lin"][t] = float(((y - pl) ** 2).mean()); mse["per"][t] = float(((y - pp) ** 2).mean())
        past = [q for q in range(max(P.ORIGINS[0], t - K), t)]          # origins q whose target q+1 <= t is observed
        fb = len(past) >= 3 and np.mean([mse["lin"][q] for q in past]) > (1 + D) * np.mean([mse["per"][q] for q in past])
        pf = pp if fb else pl; fbn[ph][0] += int(fb); fbn[ph][1] += 1
        for k, p in (("lin", pl), ("per", pp), ("fb", pf)):
            i = 0 if ph == "pre" else 1; out[k][i] += float(((y - p) ** 2).sum()); out[k][2 + i] += P.U
            c = P.cnt(p < THR, ev)
            for j in range(4): cnt[f"{k}|{ph}"][j] += c[j]
    return {"sse": out, "counts": cnt, "fb_rate": {k: v[0] / v[1] for k, v in fbn.items()}}
def cost(c, w=3): return w * c[2] + c[1]
if MODE == "tune":
    res = {}
    for K, D in itertools.product((3, 6, 12), (0.0, 0.05, 0.10)):
        tot = 0
        for sc in ("regime", "linear"):
            P = load(sc)
            for r in range(10):
                x = rep(P, 900000 + r, K, D)
                tot += sum(cost(x["counts"][f"fb|{ph}"]) - cost(x["counts"][f"lin|{ph}"]) for ph in ("pre", "post"))
        res[f"{K}|{D}"] = tot / 20; print(K, D, round(tot / 20, 2), flush=True)
    best = min(res, key=res.get); K, D = best.split("|")
    json.dump({"K": int(K), "DELTA": float(D), "grid": res}, open(f"{OUT}/h9_params.json", "w"), indent=1); print("chosen", best)
else:
    prm = json.load(open(f"{OUT}/h9_params.json")); K, D = prm["K"], prm["DELTA"]; out = {"K": K, "DELTA": D}
    for sc in ("linear", "nonlinear", "hetero", "regime", "combined"):
        P = load(sc); out[sc] = [rep(P, 20260 + r, K, D) for r in range(20)]; print(sc, flush=True)
    json.dump(out, open(f"{OUT}/h9_raw.json", "w"))
