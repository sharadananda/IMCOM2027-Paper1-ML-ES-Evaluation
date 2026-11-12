import json, numpy as np
rng=np.random.default_rng(7); B=2000
D=json.load(open("explain_raw.json")); S={}
def ci(a):
    a=np.asarray(a,float); n=len(a); bs=[a[rng.integers(0,n,n)].mean() for _ in range(B)]; return [float(a.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
for sc,R in D.items():
    o={}
    # (a) all three drivers incl. inertia
    Ra=[r for r in R if r["n"]>0]; o["n_alerts_tp"]=int(sum(r["n"] for r in Ra))
    o["true_driver_share"]=[float(x) for x in np.sum([r["true_counts"] for r in Ra],0)/o["n_alerts_tp"]]
    o["rule_fires_on_every_inertia_case"]=float(sum(r["by_driver"]["0"]["rule"] for r in Ra)/sum(r["by_driver"]["0"]["n"] for r in Ra))
    o["inertia_named_by_models_when_true_ctrl"]=float(sum(r["by_driver"][str(g)]["lin"] for r in Ra for g in (1,2))/sum(r["by_driver"][str(g)]["n"] for r in Ra for g in (1,2)))
    # (b) controllable lever (cycle time vs utilization)
    Rc=[r for r in R if r["ctrl"]["n"]>0]; c=lambda k:[r["ctrl"][k]/r["ctrl"]["n"] for r in Rc]
    o["ctrl_n"]=int(sum(r["ctrl"]["n"] for r in Rc))
    o["cycle_share"]=ci(c("cyc_true")); o["linear"]=ci(c("lin")); o["rf"]=ci(c("rf")); o["random"]=ci(c("rand")); o["rule_hit"]=ci(c("rule_hit")); o["rule_size"]=ci(c("rule_size")); o["rule_chance"]=ci(c("rule_chance"))
    o["rule_minus_chance"]=ci([r["ctrl"]["rule_hit"]/r["ctrl"]["n"]-r["ctrl"]["rule_chance"]/r["ctrl"]["n"] for r in Rc])
    o["linear_minus_rule"]=ci([(r["ctrl"]["lin"]-r["ctrl"]["rule_hit"])/r["ctrl"]["n"] for r in Rc])
    o["linear_minus_majority"]=ci([r["ctrl"]["lin"]/r["ctrl"]["n"]-max(r["ctrl"]["cyc_true"],r["ctrl"]["n"]-r["ctrl"]["cyc_true"])/r["ctrl"]["n"] for r in Rc])
    o["rf_minus_linear"]=ci([(r["ctrl"]["rf"]-r["ctrl"]["lin"])/r["ctrl"]["n"] for r in Rc])
    S[sc]=o
json.dump(S,open("explain_summary.json","w"),indent=1)
f=lambda x:f"{x[0]:.2f} [{x[1]:.2f},{x[2]:.2f}]"
for sc,o in S.items(): print(sc,"rule-chance",f(o["rule_minus_chance"]),"| linear-rule",f(o["linear_minus_rule"]),"| linear-majority",f(o["linear_minus_majority"]),"| rf-linear",f(o["rf_minus_linear"]),"| inertia-named",round(o["inertia_named_by_models_when_true_ctrl"],2),"rule fires on inertia",round(o["rule_fires_on_every_inertia_case"],2))
