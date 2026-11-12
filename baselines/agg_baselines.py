import json, glob, numpy as np
rng=np.random.default_rng(7); B=2000
M=["persistence","linear","ses","ar1_unit","fe_linear","rf","gb","rf_cv","gb_cv"]; S={}
def ci(a):
    a=np.asarray(a,float); n=len(a); bs=[a[rng.integers(0,n,n)].mean() for _ in range(B)]; return [float(a.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
for sc in ("linear","nonlinear","hetero","regime","combined"):
    R=[json.load(open(f)) for f in sorted(glob.glob(f"bl_{sc}_*.json"))]
    rm=lambda r,m:np.sqrt((r["sse"][m][0]+r["sse"][m][1])/(r["sse"][m][2]+r["sse"][m][3]))
    o={"R":len(R)}
    for m in M: o[m]={"rmse":ci([rm(r,m) for r in R]),"d_vs_linear":ci([rm(r,m)-rm(r,"linear") for r in R]),"d_vs_persistence":ci([rm(r,m)-rm(r,"persistence") for r in R])}
    S[sc]=o
json.dump(S,open("baselines_summary.json","w"),indent=1)
f=lambda x:f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"
for sc,o in S.items():
    print(sc,"R",o["R"],"persistence RMSE %.2f linear %.2f"%(o["persistence"]["rmse"][0],o["linear"]["rmse"][0]))
    print("   vs linear:",{m:f(o[m]["d_vs_linear"]) for m in M if m not in("linear",)})
