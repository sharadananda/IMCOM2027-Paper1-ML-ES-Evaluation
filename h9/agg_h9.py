import json, numpy as np
rng=np.random.default_rng(7); B=2000
D=json.load(open("h9_raw.json")); K,DL=D["K"],D["DELTA"]
def ci(a):
    a=np.asarray(a,float); n=len(a); bs=[a[rng.integers(0,n,n)].mean() for _ in range(B)]
    return [float(a.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
cost=lambda c,w=3:w*c[2]+c[1]
S={"K":K,"DELTA":DL}
for sc in ("linear","nonlinear","hetero","regime","combined"):
    R=D[sc]; o={}
    rm=lambda r,k,i:np.sqrt(r["sse"][k][i]/r["sse"][k][2+i])
    o["post_rmse"]={k:ci([rm(r,k,1) for r in R]) for k in ("lin","per","fb")}
    o["pre_rmse"]={k:ci([rm(r,k,0) for r in R]) for k in ("lin","per","fb")}
    o["d_post_rmse_fb_minus_lin"]=ci([rm(r,"fb",1)-rm(r,"lin",1) for r in R])
    o["d_post_rmse_fb_minus_per"]=ci([rm(r,"fb",1)-rm(r,"per",1) for r in R])
    o["d_pre_rmse_fb_minus_lin"]=ci([rm(r,"fb",0)-rm(r,"lin",0) for r in R])
    for ph in ("pre","post"):
        o[f"d_cost3_{ph}_fb_minus_lin"]=ci([cost(r["counts"][f"fb|{ph}"])-cost(r["counts"][f"lin|{ph}"]) for r in R])
        o[f"fb_rate_{ph}"]=ci([r["fb_rate"][ph] for r in R])
    S[sc]=o
json.dump(S,open("h9_summary.json","w"),indent=1)
f=lambda x:f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]"
print("K",K,"delta",DL)
for sc in ("linear","nonlinear","hetero","regime","combined"):
    o=S[sc]; print(f"{sc:10s} postRMSE lin {o['post_rmse']['lin'][0]:.2f} per {o['post_rmse']['per'][0]:.2f} fb {o['post_rmse']['fb'][0]:.2f} | fb-lin post {f(o['d_post_rmse_fb_minus_lin'])} | fb-lin pre {f(o['d_pre_rmse_fb_minus_lin'])} | cost post {f(o['d_cost3_post_fb_minus_lin'])} pre {f(o['d_cost3_pre_fb_minus_lin'])} | fb rate pre {o['fb_rate_pre'][0]:.2f} post {o['fb_rate_post'][0]:.2f}")
