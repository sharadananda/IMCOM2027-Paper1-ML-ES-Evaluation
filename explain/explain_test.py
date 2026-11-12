"""H10 – explanation fidelity. Usage: python explain_test.py OUTDIR   (scenarios linear, hetero, regime; 20 replicates; seeds 20260+r)
At each origin t, for every forecast alert (linear forecast < 70) that is a true event (e_{t+1} < 70), the explainer names the driver of the shortfall using information up to t.
Ground truth: the data-generating equation e_s - mu_s = 0.55(e_{s-1}-mu_s) + slope_c(c_s-1) + w-term + noise gives three contributions (inertia, cycle time, utilization);
the true driver is the most negative one. Explainers: rule trace (pre-specified rules, may fire 0..3), linear attribution, RF occlusion, random, always-inertia.
Nonlinear scenarios are excluded because the nonlinear terms have no clean three-way split."""
import sys, json, importlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
OUT = sys.argv[1]; THR = 70.0; G = {0: [0, 1, 6], 1: [2, 3], 2: [4, 5]}   # 0 inertia, 1 cycle time, 2 utilization
def load(sc):
    sys.argv = ["x", "0", "1", ".", sc]
    import run_experiment_prospective as P; importlib.reload(P); return P
def rep(P, r):
    seed = P.BASE_SEED + r; rng = np.random.default_rng(seed); mu, c0, w0, drift, bc, bw = P.draw(rng); E, C, W = P.simulate(rng, mu, c0, w0, drift, bc, bw); cs = np.cumsum(E, 1)
    res = {"n": 0, "lin": 0, "rf": 0, "rand": 0, "inertia": 0, "rule_hit": 0, "rule_size": 0, "rule_none": 0, "true_counts": [0, 0, 0], "rule_chance": 0.0, "by_driver": {str(g): {"n": 0, "lin": 0, "rf": 0, "rule": 0} for g in range(3)}, "fire_when_not": [0, 0, 0], "ctrl": {"n": 0, "lin": 0, "rf": 0, "rule_hit": 0, "rule_size": 0, "rule_chance": 0.0, "rand": 0, "cyc_true": 0}, "not_n": [0, 0, 0]}
    rf = None; rr = np.random.default_rng(seed + 5)
    for t in P.ORIGINS:
        s = t + 1; y = E[:, s]; Xtr, ytr = P.train_set(E, C, W, cs, t); Xte = P.feats(E, C, W, cs, s); lin = LinearRegression().fit(Xtr, ytr)
        if t in P.RETRAIN_RF_GB or rf is None: rf = RandomForestRegressor(80, max_depth=5, random_state=seed, n_jobs=1).fit(Xtr, ytr)
        pl = lin.predict(Xte); sel = np.where((pl < THR) & (y < THR))[0]
        if len(sel) == 0: continue
        post = s >= P.DRIFT_ON; mu_s = mu - 5.0 * (drift & post) - (3.0 if (P.REG and post) else 0.0); slope = bc * (2.5 if (P.REG and post) else 1.0)
        contrib = np.column_stack([0.55 * (E[:, s - 1] - mu_s), slope * (C[:, s] - 1.0), bw * (W[:, s] - 0.72)])   # linear DGP only
        xbar = Xtr.mean(0); pr_rf = rf.predict(Xte)
        att_lin = np.column_stack([sum(lin.coef_[j] * (Xte[:, j] - xbar[j]) for j in G[g]) for g in range(3)])
        att_rf = np.empty((U := len(y), 3))
        for g in range(3):
            Xo = Xte.copy(); Xo[:, G[g]] = xbar[G[g]]; att_rf[:, g] = pr_rf - rf.predict(Xo)
        fired = np.column_stack([E[:, t] < 75.0, (C[:, t] > C[:, t - 1]) & (C[:, t] > 1.05), W[:, t] < 0.72])
        for u in sel:
            tr = int(np.argmin(contrib[u])); res["n"] += 1; res["true_counts"][tr] += 1
            res["lin"] += int(np.argmin(att_lin[u]) == tr); res["rf"] += int(np.argmin(att_rf[u]) == tr)
            res["rand"] += int(rr.integers(0, 3) == tr); res["inertia"] += int(tr == 0)
            bd = res["by_driver"][str(tr)]; bd["n"] += 1; bd["lin"] += int(np.argmin(att_lin[u]) == tr); bd["rf"] += int(np.argmin(att_rf[u]) == tr); bd["rule"] += int(fired[u][tr])
            for g in range(3):
                if g != tr: res["not_n"][g] += 1; res["fire_when_not"][g] += int(fired[u][g])
            if contrib[u, 1:].min() < 0:     # controllable-lever question: cycle time or utilization?
                ct = 1 + int(np.argmin(contrib[u, 1:])); c_ = res["ctrl"]; c_["n"] += 1; c_["cyc_true"] += int(ct == 1)
                c_["lin"] += int(1 + np.argmin(att_lin[u, 1:]) == ct); c_["rf"] += int(1 + np.argmin(att_rf[u, 1:]) == ct); c_["rand"] += int(1 + rr.integers(0, 2) == ct)
                kk = int(fired[u, 1:].sum()); c_["rule_size"] += kk; c_["rule_hit"] += int(fired[u, ct]); c_["rule_chance"] += kk / 2.0
            k = int(fired[u].sum()); res["rule_size"] += k; res["rule_none"] += int(k == 0); res["rule_hit"] += int(fired[u][tr]); res["rule_chance"] += k / 3.0
    return res
out = {}
for sc in ("linear", "hetero", "regime"):
    P = load(sc); out[sc] = [rep(P, r) for r in range(20)]; print(sc, flush=True)
json.dump(out, open(f"{OUT}/explain_raw.json", "w"))
