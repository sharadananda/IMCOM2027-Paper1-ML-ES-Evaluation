import json,numpy as np
rng=np.random.default_rng(5); SC=["linear","nonlinear","hetero","regime","combined"]
SIG=[1,2,4,8];BIAS=[0.0,0.25,0.5];PHI=[0.0,1.0];WS=[1,2,3,5,10]
out={}
def wstar(tp,fp): return fp/tp if tp>0 else float('nan')
for sc in SC:
    R=json.load(open(f"rs_{sc}.json")); n=len(R); o={}
    base=np.sum([r["base"] for r in R],0)
    o["alerts_per_rep"]=float((base[0]+base[1])/n); o["precision_no_review"]=float(base[0]/(base[0]+base[1]))
    for k in R[0]["rev"]:
        a=np.array([r["rev"][k] for r in R],float)  # n x 4: TPdis,FPdis,TPkept,FPkept
        ws=wstar(a[:,0].sum(),a[:,1].sum()); bs=[]
        for _ in range(2000):
            ii=rng.integers(0,n,n); s=a[ii].sum(0); bs.append(wstar(s[0],s[1]))
        bs=np.array(bs,float)
        d={w:(w*a[:,0]-a[:,1]).mean() for w in WS}   # delta cost per replicate (review - none)
        dci={w:[float(np.percentile([(w*a[ii,0]-a[ii,1]).mean() for ii in [rng.integers(0,n,n) for _ in range(500)]],p)) for p in (2.5,97.5)] for w in (3,)}
        o[k]={"wstar":[float(ws),float(np.nanpercentile(bs,2.5)),float(np.nanpercentile(bs,97.5))],"dcost":{str(w):float(v) for w,v in d.items()},"dcost3_ci":dci[3],"dismissed_tp":float(a[:,0].mean()),"dismissed_fp":float(a[:,1].mean())}
    out[sc]=o
json.dump(out,open("rs_summary.json","w"),indent=1)
for sc in SC:
    print(sc,"alerts/rep %.0f prec %.2f"%(out[sc]["alerts_per_rep"],out[sc]["precision_no_review"]))
    for k in ["1|0.0|0.0","2|0.0|0.0","4|0.0|0.0","8|0.0|0.0","4|0.5|0.0","4|0.0|1.0","4|0.5|1.0"]:
        v=out[sc][k]; print("  sigma|bias|phi",k,"w*=%.2f [%.2f,%.2f]"%tuple(v["wstar"]),"dcost@3=%.1f [%.1f,%.1f]"%(v["dcost"]["3"],*v["dcost3_ci"]))
