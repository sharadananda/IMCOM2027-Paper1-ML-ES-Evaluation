"""Supplementary Simar-Wilson (2007) Algorithm 2 second-stage check for Paper 1 (H6 context).
Usage: python sw_stage2.py R0 R1 STEP OFF   (one json per replicate, resumable; needs run_experiment_prospective.py on PYTHONPATH)
Design: linear scenario of the main study, plus two exogenous 'environment' variables NOT used in Stage 1:
  z1 ~ N(0,1) shifts a unit's baseline efficiency by +3 points (true driver), z2 ~ N(0,1) has no effect (placebo).
Stage 1: input-oriented CCR DEA, inputs = mean cycle time and mean slack (1-utilization), output = mean efficiency, months 12..23.
Stage 2: delta = 1/theta >= 1 regressed on (1, z1, z2); (a) naive OLS with conventional SEs; (b) Simar-Wilson Algorithm 2
(bias-corrected scores, truncated-normal MLE, double bootstrap with B1 bias replicates and B2 inference replicates)."""
import sys, os, json, numpy as np, warnings
from scipy.optimize import linprog, minimize
from scipy.stats import truncnorm, norm
warnings.filterwarnings("ignore")
R0, R1, STEP, OFF = [int(a) for a in sys.argv[1:5]]
B1, B2 = int(os.environ.get("B1", 40)), int(os.environ.get("B2", 400))
sys.argv = ["x", "0", "1", ".", "linear"]
import run_experiment_prospective as P
U = P.U
LO, HI = 12, 24

def theta_vs(xi, yi, Xr, Yr):
    n = Xr.shape[0]; c = np.zeros(1 + n); c[0] = 1
    A = [np.r_[-xi[k], Xr[:, k]] for k in range(Xr.shape[1])] + [np.r_[0, -Yr[:, k]] for k in range(Yr.shape[1])]
    b = [0.0] * Xr.shape[1] + [-yi[k] for k in range(Yr.shape[1])]
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)] * (1 + n), method="highs")
    return r.fun if r.success else np.nan

def dea_delta(X, Y, Xref=None):
    Xr = X if Xref is None else Xref
    return np.array([1.0 / theta_vs(X[i], Y[i], Xr, Y) for i in range(len(X))])

def trunc_reg(d, Z):
    m = np.isfinite(d) & (d > 1 + 1e-9); z, y = Z[m], d[m]
    b0 = np.linalg.lstsq(z, y, rcond=None)[0]; s0 = np.std(y - z @ b0)
    def nll(p):
        b, s = p[:-1], np.exp(p[-1]); mu = z @ b
        return -np.sum(norm.logpdf((y - mu) / s) - np.log(s) - norm.logcdf((mu - 1) / s))
    r = minimize(nll, np.r_[b0, np.log(s0)], method="L-BFGS-B")
    return r.x[:-1], float(np.exp(r.x[-1]))

def draw_delta(Z, b, s, rng):
    mu = Z @ b; a = (1 - mu) / s
    return mu + truncnorm.rvs(a, np.inf, loc=0, scale=s, random_state=rng)

def rep(r):
    rng = np.random.default_rng(20260 + r)
    mu, c0, w0, drift, bc, bw = P.draw(rng)
    z1, z2 = rng.normal(size=U), rng.normal(size=U)
    mu = np.clip(mu + 3.0 * z1, 55, 95)
    E, C, W = P.simulate(rng, mu, c0, w0, drift, bc, bw)
    X = np.column_stack([C[:, LO:HI].mean(1), 1 - W[:, LO:HI].mean(1)]); Y = E[:, LO:HI].mean(1)[:, None]
    Z = np.column_stack([np.ones(U), z1, z2])
    d = dea_delta(X, Y)
    # naive OLS
    bo = np.linalg.lstsq(Z, d, rcond=None)[0]; res = d - Z @ bo
    se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * (res @ res) / (U - 3))
    ols = [[float(bo[k]), float(bo[k] - 1.96 * se[k]), float(bo[k] + 1.96 * se[k])] for k in (1, 2)]
    # Algorithm 2
    b1, s1 = trunc_reg(d, Z)
    ds = []
    for _ in range(B1):
        dstar = draw_delta(Z, b1, s1, rng)
        ds.append(dea_delta(X * (dstar / d)[:, None], Y))
    dm = np.nanmean(ds, axis=0); dd = np.where(np.isfinite(dm), 2 * d - dm, d)
    b2, s2 = trunc_reg(dd, Z)
    bs = []
    for _ in range(B2):
        dstar = draw_delta(Z, b2, s2, rng)
        dh = dea_delta(X * (dstar / d)[:, None], Y)
        try: bs.append(trunc_reg(dh, Z)[0])
        except Exception: pass
    bs = np.array(bs) - b2
    sw = []
    for k in (1, 2):
        a, b = np.quantile(bs[:, k], [0.025, 0.975]); sw.append([float(b2[k]), float(b2[k] - b), float(b2[k] - a)])
    return {"r": r, "n_eff_units": int((d > 1 + 1e-9).sum()), "ols": ols, "sw": sw, "B1": B1, "B2": len(bs)}

if __name__ == "__main__":
    for r in range(R0 + OFF, R1, STEP):
        fn = f"sw_{r}.json"
        if os.path.exists(fn): continue
        json.dump(rep(r), open(fn, "w")); print("rep", r, flush=True)
