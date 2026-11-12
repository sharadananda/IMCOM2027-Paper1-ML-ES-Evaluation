import json,glob,numpy as np
R=[json.load(open(f)) for f in sorted(glob.glob("sw_*.json"))]; rng=np.random.default_rng(3); n=len(R)
out={"R":n,"B1":R[0]["B1"],"B2_mean":float(np.mean([r["B2"] for r in R])),"units_above_frontier_mean":float(np.mean([r["n_eff_units"] for r in R]))}
def bci(a):
    a=np.asarray(a,float);bs=[rng.choice(a,len(a)).mean() for _ in range(2000)];return [float(a.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
for m in("ols","sw"):
    for k,name in((0,"z1_true"),(1,"z2_placebo")):
        est=[r[m][k][0] for r in R];lo=[r[m][k][1] for r in R];hi=[r[m][k][2] for r in R]
        sig=[(l>0 or h<0) for l,h in zip(lo,hi)]; neg=[h<0 for h in hi]; pos=[l>0 for l in lo]
        out[f"{m}_{name}"]={"mean_est":bci(est),"sd_est":float(np.std(est)),"ci_excl_zero":float(np.mean(sig)),"neg_sig":float(np.mean(neg)),"pos_sig":float(np.mean(pos)),"mean_ci_width":float(np.mean(np.subtract(hi,lo)))}
out["width_ratio_sw_over_ols_z1"]=out["sw_z1_true"]["mean_ci_width"]/out["ols_z1_true"]["mean_ci_width"]
# correlation between OLS and SW estimates, z1
out["corr_est_z1"]=float(np.corrcoef([r["ols"][0][0] for r in R],[r["sw"][0][0] for r in R])[0,1])
json.dump(out,open("sw_summary.json","w"),indent=1);print(json.dumps(out,indent=1))
