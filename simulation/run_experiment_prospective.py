"""
Prospective (rolling-origin) synthetic experiment for Paper 1.
Usage:  python run_experiment_prospective.py START COUNT [OUTDIR] [SCENARIO]
SCENARIO in {linear, nonlinear, hetero, regime, combined}. Replicate r uses seed BASE_SEED + r.
Writes OUTDIR/prospective_raw_<SCENARIO>_<START>.json
Then run aggregate_prospective.py to bootstrap over replicates.
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")

START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = sys.argv[3] if len(sys.argv) > 3 else "/home/claude"
SCEN = sys.argv[4] if len(sys.argv) > 4 else "linear"
NL = SCEN in ("nonlinear", "combined")      # threshold + interaction + saturating terms
HET = SCEN in ("hetero", "combined")        # unit-specific slopes
REG = SCEN in ("regime", "combined")        # population-wide regime change at month 25

BASE_SEED = 20260
U, T = 60, 36                       # units, months (indices 0..35)
ORIGINS = list(range(11, 35))       # origin t: information up to month t; target month t+1 (12..35)
RETRAIN_RF_GB = set(range(11, 35, 3))
DEA_ORIGINS = list(range(11, 34, 3))
WINDOWS = [3, 6, 12, "expanding"] if SCEN == "linear" else [6]
THR = 70.0                          # adverse event: efficiency index < 70
DRIFT_ON = 25                       # months >= 25: drift for flagged units
THRS = [64, 66, 68, 70, 72, 74, 76, 78]
STAMPS = [0.0, 0.5]
SIGMAS = [1, 2, 3, 4, 6, 8, 12]
CN = ["linear", "ar1", "rf", "gb", "persistence", "unit_mean", "roll3", "static_linear"]


def draw(rng):
    mu = np.clip(rng.normal(78, 6, U), 60, 92)
    c0 = np.clip(rng.normal(1.05, 0.12, U), 0.8, 1.4)
    w0 = np.clip(rng.normal(0.72, 0.08, U), 0.5, 0.95)
    drift = rng.random(U) < 0.30
    bc = rng.normal(-6.0, 3.0, U) if HET else np.full(U, -6.0)
    bw = rng.normal(10.0, 4.0, U) if HET else np.full(U, 10.0)
    return mu, c0, w0, drift, bc, bw


def simulate(rng, mu, c0, w0, drift, bc, bw):
    E = np.zeros((U, T)); C = np.zeros((U, T)); W = np.zeros((U, T))
    e_prev, c_prev, w_prev = mu.copy(), c0.copy(), w0.copy()
    for t in range(T):
        c = np.clip(c_prev + rng.normal(0, 0.03, U), 0.8, 1.4)
        w = np.clip(w_prev + rng.normal(0, 0.02, U), 0.5, 0.95)
        mu_t = mu - 5.0 * (drift & (t >= DRIFT_ON))
        slope_c = bc * (2.5 if (REG and t >= DRIFT_ON) else 1.0)
        if REG and t >= DRIFT_ON: mu_t = mu_t - 3.0
        wterm = 12.0 * np.tanh((w - 0.72) / 0.10) if NL else bw * (w - 0.72)
        nl = (-25.0 * np.maximum(c - 1.10, 0) + 150.0 * (c - 1.0) * (w - 0.72)) if NL else 0.0
        e = np.clip(0.55 * e_prev + 0.45 * mu_t + slope_c * (c - 1.0) + wterm + nl
                    + rng.normal(0, 1.8, U), 50, 95)
        E[:, t], C[:, t], W[:, t] = e, c, w
        e_prev, c_prev, w_prev = e, c, w
    return E, C, W


def feats(E, C, W, cs, s):
    """Features for target month s (only months <= s-1 used)."""
    return np.column_stack([E[:, s-1], E[:, s-2], C[:, s-1], C[:, s-2],
                            W[:, s-1], W[:, s-2], cs[:, s-1] / s])


def train_set(E, C, W, cs, t):
    X = np.vstack([feats(E, C, W, cs, s) for s in range(2, t + 1)])
    y = np.concatenate([E[:, s] for s in range(2, t + 1)])
    return X, y


def ccr(x, y, i):
    n = x.shape[0]
    c = np.zeros(1 + n); c[0] = 1
    A, b = [], []
    for k in range(x.shape[1]):
        r = np.zeros(1 + n); r[0] = -x[i, k]; r[1:] = x[:, k]; A.append(r); b.append(0)
    for k in range(y.shape[1]):
        r = np.zeros(1 + n); r[1:] = -y[:, k]; A.append(r); b.append(-y[i, k])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * (1 + n), method="highs")
    return res.fun if res.success else np.nan


def dea_scores(E, C, W, lo, hi):
    """CCR input-oriented DEA on months lo..hi-1 (inclusive lo, exclusive hi)."""
    x = np.column_stack([C[:, lo:hi].mean(1), 1 - W[:, lo:hi].mean(1)])
    y = E[:, lo:hi].mean(1)[:, None]
    return np.array([ccr(x, y, i) for i in range(U)])


def cnt(alert, ev):
    return [int((alert & ev).sum()), int((alert & ~ev).sum()),
            int((~alert & ev).sum()), int((~alert & ~ev).sum())]


def add(d, k, v):
    if k not in d: d[k] = [0, 0, 0, 0]
    for j in range(4): d[k][j] += v[j]


def r2_ols(X, y):
    m = LinearRegression().fit(X, y)
    return m.score(X, y)


def one_replicate(r):
    seed = BASE_SEED + r
    rng = np.random.default_rng(seed)
    mu, c0, w0, drift, bc, bw = draw(rng)
    E, C, W = simulate(rng, mu, c0, w0, drift, bc, bw)
    hold = np.zeros(U, bool); hold[np.random.default_rng(seed + 7).choice(U, 12, replace=False)] = True
    cs = np.cumsum(E, axis=1)
    rec = {"seed": seed}
    sse = {m: [0.0, 0.0, 0, 0, 0.0] for m in CN}   # sse_pre, sse_post, n_pre, n_post, sae
    ytot = []
    counts, counts_phase = {}, {}
    sweep, review = {}, {}
    rf = gb = None
    ssg = {m: [0.0, 0] for m in ("linear", "rf", "gb", "persistence", "unit_mean")}   # unit-holdout
    rfg = gbg = None
    fc, dscore = {}, {}
    static = None
    rrng = np.random.default_rng(seed + 999)
    for t in ORIGINS:
        s = t + 1
        y = E[:, s]; ev = y < THR
        post = s >= DRIFT_ON
        Xtr, ytr = train_set(E, C, W, cs, t)
        Xte = feats(E, C, W, cs, s)
        lin = LinearRegression().fit(Xtr, ytr)
        ar = LinearRegression().fit(Xtr[:, [0]], ytr)
        if static is None:
            static = LinearRegression().fit(Xtr, ytr)      # trained once at first origin
        if t in RETRAIN_RF_GB or rf is None:
            rf = RandomForestRegressor(80, max_depth=5, random_state=seed, n_jobs=1).fit(Xtr, ytr)
            gb = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08,
                                           random_state=seed).fit(Xtr, ytr)
        lag = E[:, t]
        r3 = E[:, t-2:t+1].mean(1)
        P = {"linear": lin.predict(Xte), "ar1": ar.predict(Xte[:, [0]]), "rf": rf.predict(Xte),
             "gb": gb.predict(Xte), "persistence": lag, "unit_mean": cs[:, t] / (t + 1),
             "roll3": r3, "static_linear": static.predict(Xte)}
        for m, p in P.items():
            e2 = float(((y - p) ** 2).sum())
            sse[m][0 if not post else 1] += e2
            sse[m][2 if not post else 3] += U
            sse[m][4] += float(np.abs(y - p).sum())
        ytot.append(y)
        fc[t] = P["linear"].copy()
        # group (unit) hold-out: train on 48 units only, predict 12 unseen units
        tr = ~hold
        Xg = np.vstack([feats(E, C, W, cs, q)[tr] for q in range(2, t + 1)])
        yg = np.concatenate([E[tr, q] for q in range(2, t + 1)])
        lg = LinearRegression().fit(Xg, yg)
        if t in RETRAIN_RF_GB or rfg is None:
            rfg = RandomForestRegressor(80, max_depth=5, random_state=seed, n_jobs=1).fit(Xg, yg)
            gbg = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=seed).fit(Xg, yg)
        PG = {"linear": lg.predict(Xte[hold]), "rf": rfg.predict(Xte[hold]), "gb": gbg.predict(Xte[hold]),
              "persistence": lag[hold] if False else E[hold, t], "unit_mean": cs[hold, t] / (t + 1)}
        for m, p in PG.items():
            ssg[m][0] += float(((y[hold] - p) ** 2).sum()); ssg[m][1] += int(hold.sum())
        cyc = C[:, t] > C[:, t - 1]
        pol = {"lag_threshold": lag < THR, "roll3_threshold": r3 < THR,
               "cycle_only": cyc,
               "forecast_linear": P["linear"] < THR, "forecast_gb": P["gb"] < THR,
               "forecast_static": P["static_linear"] < THR,
               "forecast_linear_AND_cycle": (P["linear"] < THR) & cyc,
               "forecast_gb_AND_cycle": (P["gb"] < THR) & cyc,
               "lag_AND_cycle": (lag < THR) & cyc}
        for k, a in pol.items():
            v = cnt(a, ev); add(counts, k, v)
            if k in ("lag_threshold", "forecast_linear", "forecast_static"):
                add(counts_phase, f"{k}|{'post' if post else 'pre'}", v)
        # threshold sweep
        for th in THRS:
            add(sweep, f"lag|{th}", cnt(lag < th, ev))
            add(sweep, f"forecast_linear|{th}", cnt(P["linear"] < th, ev))
            add(sweep, f"forecast_gb|{th}", cnt(P["gb"] < th, ev))
            add(sweep, f"forecast_linear_AND_cycle|{th}", cnt((P["linear"] < th) & cyc, ev))
        # simulated reviewer (stylised): sees true outcome + N(0,sigma); may only dismiss alerts
        z = rrng.normal(0, 1, U); uu = rrng.random(U)
        for pk in ("forecast_linear", "forecast_linear_AND_cycle"):
            a = pol[pk]
            for st in STAMPS:
                for sg in SIGMAS:
                    est = y + sg * z
                    kept = a & ((uu < st) | (est < THR))
                    add(review, f"{pk}|{st}|{sg}", cnt(kept, ev))
    ally = np.concatenate(ytot)
    rec["sst"] = float(((ally - ally.mean()) ** 2).sum()); rec["n"] = int(len(ally))
    rec["sse"] = sse; rec["counts"] = counts; rec["counts_phase"] = counts_phase
    rec["sweep"] = sweep; rec["review"] = review; rec["sse_group"] = ssg; rec["scenario"] = SCEN
    rec["n_drift_units"] = int(drift.sum())
    rec["event_rate"] = float(np.mean(ally < THR))
    # ---- DEA with aligned windows, only data before each origin
    dea = {}
    for w in WINDOWS:
        rho, rho_lag, d2, cov, rho_fc = [], [], [], [], []
        sc_list, fc_list = [], []
        for t in DEA_ORIGINS:
            lo = 0 if w == "expanding" else max(0, t + 1 - w)
            sc = dea_scores(E, C, W, lo, t + 1)
            ok = ~np.isnan(sc); cov.append(float(ok.mean()))
            sc_list.append(sc); fc_list.append(fc[t])
            if ok.sum() > 5: rho_fc.append(spearmanr(sc[ok], fc[t][ok])[0])
            y = E[:, t + 1]; lag = E[:, t]
            if ok.sum() > 5:
                rho.append(spearmanr(sc[ok], y[ok])[0]); rho_lag.append(spearmanr(lag[ok], y[ok])[0])
                b = r2_ols(lag[ok][:, None], y[ok])
                f = r2_ols(np.column_stack([lag[ok], sc[ok]]), y[ok]); d2.append(f - b)
        dea[str(w)] = {"rho": float(np.mean(rho)), "rho_lag": float(np.mean(rho_lag)),
                       "dR2": float(np.mean(d2)), "coverage": float(np.mean(cov)),
                       "rho_dea_fc": float(np.mean(rho_fc)), "rho_dea_fc_sd": float(np.std(rho_fc)),
                       "stab_dea": float(np.mean([spearmanr(a, b, nan_policy="omit")[0] for a, b in zip(sc_list[:-1], sc_list[1:])])),
                       "stab_fc": float(np.mean([spearmanr(a, b)[0] for a, b in zip(fc_list[:-1], fc_list[1:])]))}
    # leaky version: DEA on the whole panel (incl. months after the origin) -- the earlier design
    sc = dea_scores(E, C, W, 0, T)
    rho, d2 = [], []
    for t in DEA_ORIGINS:
        y = E[:, t + 1]; lag = E[:, t]
        rho.append(spearmanr(sc, y)[0])
        d2.append(r2_ols(np.column_stack([lag, sc]), y) - r2_ols(lag[:, None], y))
    dea["leaky_full_period"] = {"rho": float(np.mean(rho)), "dR2": float(np.mean(d2)), "coverage": 1.0}
    rec["dea"] = dea
    return rec


if __name__ == "__main__":
    recs = []
    for r in range(START, START + COUNT):
        recs.append(one_replicate(r)); print("rep", r, flush=True)
    json.dump({"start": START, "count": COUNT, "recs": recs}, open(f"{OUT}/prospective_raw_{SCEN}_{START}.json", "w"))
