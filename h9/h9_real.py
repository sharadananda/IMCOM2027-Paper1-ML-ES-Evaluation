"""H9 on DOSM data: same fallback rule (K,DELTA from simulation tuning), block bootstrap over origins."""
import sys, json, numpy as np
sys.argv=["x","qtr.csv"]; src=open("run_real_data.py").read().split("MODELS=")[0]; exec(src)
from sklearn.linear_model import LinearRegression
prm=json.load(open("h9_params.json")); out={}
for K in (prm["K"],6):
    D=prm["DELTA"]; ml={}; mp={}; rows=[]; fbr=[]
    for t in ORIG:
        s=t+1; y=G[:,s]; ev=y<EV; X,yy=trainset(t); pl=LinearRegression().fit(X,yy).predict(feats(s)); pp=G[:,t]
        ml[t]=((y-pl)**2).mean(); mp[t]=((y-pp)**2).mean(); past=[q for q in range(max(ORIG[0],t-K),t)]
        fb=len(past)>=3 and np.mean([ml[q] for q in past])>(1+D)*np.mean([mp[q] for q in past]); fbr.append(int(fb)); pf=pp if fb else pl
        r=[]
        for p in (pl,pp,pf):
            a=p<EV; r+= [((y-p)**2).sum(), int((~a&ev).sum()), int((a&~ev).sum())]
        rows.append(r)
    A=np.array(rows,float); n=len(ORIG)*U; rb=np.random.default_rng(5); nO=len(ORIG)
    def rs(): 
        st=rb.integers(0,nO,int(np.ceil(nO/4))); return np.concatenate([(x+np.arange(4))%nO for x in st])[:nO]
    IDX=[rs() for _ in range(2000)]
    def ci(f): 
        pt=f(np.arange(nO)); bs=[f(i) for i in IDX]; return [float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    rm=lambda i,c:np.sqrt(A[i,c].sum()/(len(i)*U)); cst=lambda i,c:(3*A[i,c+1]+A[i,c+2]).sum()
    out[f"K{K}"]={"fallback_rate":float(np.mean(fbr)),"rmse_lin":rm(np.arange(nO),0),"rmse_per":rm(np.arange(nO),3),"rmse_fb":rm(np.arange(nO),6),
      "d_rmse_fb_minus_lin":ci(lambda i:rm(i,6)-rm(i,0)),"d_rmse_fb_minus_per":ci(lambda i:rm(i,6)-rm(i,3)),
      "d_cost3_fb_minus_lin":ci(lambda i:cst(i,6)-cst(i,0)),"d_cost3_fb_minus_per":ci(lambda i:cst(i,6)-cst(i,3)),
      "fallback_by_year":{y:float(np.mean([f for f,t in zip(fbr,ORIG) if dates[t+1][:4]==y])) for y in sorted({dates[t+1][:4] for t in ORIG})}}
json.dump(out,open("h9_real.json","w"),indent=1)
for k,v in out.items(): print(k,json.dumps(v,default=float)[:900])
