"""Option 2: override-log reconstructability and tamper-evidence test (design-conformance check, not behavioural evidence).
Usage: PYTHONPATH=.. python audit_log_test.py [R=20]
Logs every alert (linear forecast < 70) with the reviewer's keep/dismiss decision (sigma=2, rubber-stamp prob 0.25).
An independent auditor sees ONLY the log and a model registry (coefficients per model id), never the raw data store.
Reconstruction levels per record:  L1 forecast recomputed from logged features + model id; L2 recommendation recoverable;
L3 pre-review (counterfactual) recommendation recoverable; L4 every dismissal has reason code, reviewer id and timestamp;
L5 dismissal decision itself recoverable.   Ablations remove one field family at a time.
Tamper test: modify forecast / features / decision / reason, delete, reorder, insert forged record, truncate tail."""
import sys, json, hashlib, copy, numpy as np
R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sys.argv = ["x", "0", "1", ".", "linear"]
import run_experiment_prospective as P
from sklearn.linear_model import LinearRegression
THR = P.THR
def h(rec, prev): return hashlib.sha256((json.dumps(rec, sort_keys=True) + prev).encode()).hexdigest()
def build(r):
    seed = P.BASE_SEED + r; rng = np.random.default_rng(seed); rr = np.random.default_rng(seed + 777)
    mu, c0, w0, dr, bc, bw = P.draw(rng); E, C, W = P.simulate(rng, mu, c0, w0, dr, bc, bw); cs = np.cumsum(E, axis=1)
    registry, log, prev = {}, [], "GENESIS"
    for t in P.ORIGINS:
        s = t + 1; y = E[:, s]
        Xtr, ytr = P.train_set(E, C, W, cs, t); m = LinearRegression().fit(Xtr, ytr)
        mid = f"r{r}-t{t}"; registry[mid] = {"coef": m.coef_.tolist(), "b": float(m.intercept_)}
        X = P.feats(E, C, W, cs, s); f = m.predict(X); cyc = C[:, t] > C[:, t - 1]
        for i in np.flatnonzero(f < THR):
            checked = rr.random() >= 0.25
            dismiss = bool(checked and (y[i] + 2.0 * rr.normal()) >= THR)
            rec = {"rep": r, "origin": t, "unit": int(i), "model_id": mid, "features": [float(v) for v in X[i]],
                   "forecast": float(f[i]), "threshold": THR, "rule_flags": {"forecast_below": True, "cycle_rising": bool(cyc[i])},
                   "recommendation": "alert", "reviewer_decision": "dismiss" if dismiss else "keep",
                   "reason_code": "R-FALSE-ALARM" if dismiss else None, "reviewer_id": "rev-01" if dismiss else None,
                   "ts": f"2027-01-{(t % 28) + 1:02d}T09:{i % 60:02d}:00Z" if dismiss else None}
            hh = h(rec, prev); log.append({**rec, "prev_hash": prev, "hash": hh}); prev = hh
    return log, registry
DROP = {"full": [], "no_features": ["features"], "no_model_id": ["model_id"], "no_forecast": ["forecast"],
        "no_threshold": ["threshold"], "no_recommendation": ["recommendation"], "no_decision": ["reviewer_decision"],
        "no_reason_or_reviewer": ["reason_code", "reviewer_id", "ts"],
        "minimal(unit,origin,decision)": ["features", "model_id", "forecast", "threshold", "rule_flags", "recommendation", "reason_code", "reviewer_id", "ts"]}
def reconstruct(rec, reg):
    L = {}
    f = None
    if "features" in rec and "model_id" in rec and rec["model_id"] in reg:
        m = reg[rec["model_id"]]; f = float(np.dot(m["coef"], rec["features"]) + m["b"])
    L["L1_forecast"] = f is not None and "forecast" in rec and abs(f - rec["forecast"]) < 1e-9
    fv = f if f is not None else rec.get("forecast")
    thr = rec.get("threshold")
    rec_ok = fv is not None and thr is not None
    L["L2_recommendation"] = rec_ok
    L["L3_counterfactual"] = rec_ok or ("recommendation" in rec)
    dec = rec.get("reviewer_decision")
    L["L5_decision"] = dec in ("keep", "dismiss")
    L["L4_accountability"] = (dec == "keep") or (dec == "dismiss" and all(rec.get(k) for k in ("reason_code", "reviewer_id", "ts")))
    return L
def verify_chain(log, anchor=None):
    prev = "GENESIS"
    for rec in log:
        core = {k: v for k, v in rec.items() if k not in ("prev_hash", "hash")}
        if rec["prev_hash"] != prev or h(core, prev) != rec["hash"]: return False
        prev = rec["hash"]
    return anchor is None or prev == anchor
def recompute_check(log, reg):
    for rec in log:
        m = reg[rec["model_id"]]
        if abs(np.dot(m["coef"], rec["features"]) + m["b"] - rec["forecast"]) > 1e-9: return False
        if (rec["forecast"] < rec["threshold"]) != (rec["recommendation"] == "alert"): return False
        if rec["reviewer_decision"] == "dismiss" and not (rec["reason_code"] and rec["reviewer_id"] and rec["ts"]): return False
    return True
if __name__ == "__main__":
    res = {k: {l: [] for l in ("L1_forecast", "L2_recommendation", "L3_counterfactual", "L4_accountability", "L5_decision")} for k in DROP}
    size = {"full": [], "minimal(unit,origin,decision)": []}
    tamper = {}; rng = np.random.default_rng(11); n_log = 0; n_dis = 0
    for r in range(R):
        log, reg = build(r); n_log += len(log); n_dis += sum(x["reviewer_decision"] == "dismiss" for x in log)
        for k, drop in DROP.items():
            outs = [reconstruct({a: b for a, b in rec.items() if a not in drop}, reg) for rec in log]
            for l in res[k]: res[k][l].append(float(np.mean([o[l] for o in outs])))
        for k in size:
            size[k].append(float(np.mean([len(json.dumps({a: b for a, b in rec.items() if a not in DROP[k]})) for rec in log])))
        anchor = log[-1]["hash"]
        for kind in ("modify_forecast", "modify_features", "modify_decision", "modify_reason", "delete_record", "reorder", "insert_forged", "truncate_tail"):
            for _ in range(10):
                L2 = copy.deepcopy(log); i = int(rng.integers(1, len(L2) - 2))
                if kind == "modify_forecast": L2[i]["forecast"] += 0.5
                elif kind == "modify_features": L2[i]["features"][0] += 1.0
                elif kind == "modify_decision":
                    L2[i]["reviewer_decision"] = "keep" if L2[i]["reviewer_decision"] == "dismiss" else "dismiss"
                elif kind == "modify_reason":
                    j = next((q for q in range(i, len(L2)) if L2[q]["reviewer_decision"] == "dismiss"), i); L2[j]["reason_code"] = "R-OTHER"
                elif kind == "delete_record": del L2[i]
                elif kind == "reorder": L2[i], L2[i + 1] = L2[i + 1], L2[i]
                elif kind == "insert_forged":
                    f = copy.deepcopy(L2[i]); f["reviewer_decision"] = "dismiss"; f["reason_code"] = "R-FALSE-ALARM"; f["reviewer_id"] = "rev-99"; f["ts"] = "2027-01-01T00:00:00Z"; L2.insert(i, f)
                elif kind == "truncate_tail": L2 = L2[:-int(rng.integers(1, 5))]
                d = tamper.setdefault(kind, {"recompute_only": [], "chain_no_anchor": [], "chain_anchored": []})
                d["recompute_only"].append(not recompute_check(L2, reg))
                d["chain_no_anchor"].append(not verify_chain(L2))
                d["chain_anchored"].append(not verify_chain(L2, anchor))
    out = {"R": R, "records": n_log, "dismissals": n_dis,
           "reconstruction": {k: {l: float(np.mean(v)) for l, v in d.items()} for k, d in res.items()},
           "bytes_per_record": {k: float(np.mean(v)) for k, v in size.items()},
           "tamper_detection": {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in tamper.items()}}
    json.dump(out, open("audit_summary.json", "w"), indent=1)
    print(json.dumps(out["reconstruction"], indent=0)[:1800]); print(out["bytes_per_record"]); print(json.dumps(out["tamper_detection"], indent=0))
