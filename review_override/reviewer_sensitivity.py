"""Option 1: reviewer-parameter sensitivity / break-even analysis for H8 (simulated, stylised).
Usage: PYTHONPATH=.. python reviewer_sensitivity.py SCENARIO [R=20]
Reuses the main study's generator and per-origin linear forecaster (alerts: forecast < 70).
A reviewer inspects each alert and may DISMISS it (never create one). Reviewer signal = true outcome + N(0, sigma_k^2);
dismiss if signal >= 70. Automation bias b: with prob. b the alert is rubber-stamped (kept without checking).
Fatigue phi: sigma_k = sigma * (1 + phi * k/n) for the k-th of n alerts in that origin's queue.
Dismissing a true alert converts TP->FN (cost w each); dismissing a false alert converts FP->TN (saves 1 each).
Review therefore lowers cost iff w < w* = (false alerts dismissed)/(true alerts dismissed)."""
import sys, json, numpy as np
scen = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sys.argv = ["x", "0", "1", ".", scen]
import run_experiment_prospective as P
from sklearn.linear_model import LinearRegression
SIG = [1, 2, 4, 8]; BIAS = [0.0, 0.25, 0.5]; PHI = [0.0, 1.0]; WS = [1, 2, 3, 5, 10]
def one(r):
    seed = P.BASE_SEED + r; rng = np.random.default_rng(seed)
    mu, c0, w0, dr, bc, bw = P.draw(rng); E, C, W = P.simulate(rng, mu, c0, w0, dr, bc, bw)
    cs = np.cumsum(E, axis=1); rr = np.random.default_rng(seed + 4242)
    out = {f"{s}|{b}|{f}": [0, 0, 0, 0] for s in SIG for b in BIAS for f in PHI}  # TPdis, FPdis, TPkept, FPkept
    base = [0, 0, 0, 0]  # TP FP FN TN, no review
    for t in P.ORIGINS:
        s = t + 1; y = E[:, s]; ev = y < P.THR
        Xtr, ytr = P.train_set(E, C, W, cs, t); m = LinearRegression().fit(Xtr, ytr)
        a = m.predict(P.feats(E, C, W, cs, s)) < P.THR
        c = P.cnt(a, ev)
        for j in range(4): base[j] += c[j]
        idx = np.flatnonzero(a); n = len(idx)
        if n == 0: continue
        z = rr.normal(size=n); u = rr.random(n); k = np.arange(n) / max(n, 1)
        for sg in SIG:
            for b in BIAS:
                for f in PHI:
                    est = y[idx] + sg * (1 + f * k) * z
                    dis = (u >= b) & (est >= P.THR)        # dismissed only if actually checked
                    o = out[f"{sg}|{b}|{f}"]
                    o[0] += int((dis & ev[idx]).sum()); o[1] += int((dis & ~ev[idx]).sum())
                    o[2] += int((~dis & ev[idx]).sum()); o[3] += int((~dis & ~ev[idx]).sum())
    return {"base": base, "rev": out}
if __name__ == "__main__":
    recs = []
    for r in range(R): recs.append(one(r)); print("rep", r, flush=True)
    json.dump(recs, open(f"rs_{scen}.json", "w"))
