"""Real-data pilot (secondary data): DOSM quarterly productivity by sector (CC BY 4.0).
Same prospective protocol as the simulation: rolling origin, train only on quarters <= origin, forecast next-quarter output-per-hour YoY growth.
Event = next-quarter growth < 0. Units = 19 leaf sectors/sub-sectors. Bootstrap = moving blocks of 4 origins.  Usage: python run_real_data.py qtr.csv -> real_results.json"""
import sys, json, warnings, numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")
CSV=sys.argv[1] if len(sys.argv)>1 else "qtr.csv"
q=pd.read_csv(CSV); LEAF=[s for s in sorted(q.sector.unique()) if s not in ("p0","p3","p5")]
def piv(series,col): 
    x=q[(q.series==series)&(q.sector.isin(LEAF))].pivot(index="sector",columns="date",values=col); return x.loc[LEAF]
G=piv("growth_yoy","output_hour").to_numpy(); GD=piv("growth_yoy","gdp").to_numpy(); HG=piv("growth_yoy","hours").to_numpy(); EG=piv("growth_yoy","employment").to_numpy()
LV={c:piv("abs",c).to_numpy() for c in ("gdp","hours","employment")}   # levels, 46 quarters (growth starts 4 quarters later)
dates=list(piv("growth_yoy","output_hour").columns); U,T=G.shape; OFF=LV["gdp"].shape[1]-T
ORIG=list(range(12,T-1)); RETRAIN=set(range(12,T-1,3)); EV=0.0
CS=np.cumsum(G,1)
def feats(s): return np.column_stack([G[:,s-1],G[:,s-2],GD[:,s-1],HG[:,s-1],EG[:,s-1],HG[:,s-2],CS[:,s-1]/s])
def trainset(t,units=None):
    X=np.vstack([feats(s) for s in range(2,t+1)]); y=np.concatenate([G[:,s] for s in range(2,t+1)])
    if units is not None: 
        m=np.tile(np.isin(np.arange(U),units),t-1); X,y=X[m],y[m]
    return X,y
def ccr(x,y,i):
    n=x.shape[0]; c=np.zeros(1+n); c[0]=1; A=[];b=[]
    for k in range(x.shape[1]): r=np.zeros(1+n); r[0]=-x[i,k]; r[1:]=x[:,k]; A.append(r); b.append(0)
    for k in range(y.shape[1]): r=np.zeros(1+n); r[1:]=-y[:,k]; A.append(r); b.append(-y[i,k])
    res=linprog(c,A_ub=A,b_ub=b,bounds=[(0,None)]*(1+n),method="highs"); return res.fun if res.success else np.nan
MODELS=["persistence","unit_mean","roll3","ar1","linear","rf","gb"]; THS=[-2,-1,-0.5,0,0.5,1,2,3]
sse={m:np.zeros(len(ORIG)) for m in MODELS}; sae={m:np.zeros(len(ORIG)) for m in MODELS}; sst=np.zeros(len(ORIG)); nn=np.zeros(len(ORIG))
POL=["lag_threshold","roll3_threshold","forecast_linear","forecast_gb","hours_rising_only","forecast_linear_AND_hours","lag_AND_hours"]
cnt={p:np.zeros((len(ORIG),4)) for p in POL}; swp={f"{f}|{th}":np.zeros((len(ORIG),4)) for f in ("lag","forecast_linear","forecast_linear_AND_hours") for th in THS}
# unseen-unit (5-fold by sector)
rng=np.random.default_rng(11); perm=rng.permutation(U); FOLDS=np.array_split(perm,5)
gsse={m:np.zeros(len(ORIG)) for m in ("persistence","linear","rf","gb")}; gn=np.zeros(len(ORIG))
rf=gb=None; grf={}; ggb={}
def C(a,e): return [int((a&e).sum()),int((a&~e).sum()),int((~a&e).sum()),int((~a&~e).sum())]
for k,t in enumerate(ORIG):
    s=t+1; y=G[:,s]; ev=y<EV; X,yy=trainset(t); Xte=feats(s)
    lin=LinearRegression().fit(X,yy); ar=LinearRegression().fit(X[:,[0]],yy)
    if t in RETRAIN or rf is None:
        rf=RandomForestRegressor(80,max_depth=5,random_state=0,n_jobs=1).fit(X,yy); gb=GradientBoostingRegressor(n_estimators=80,max_depth=3,learning_rate=0.08,random_state=0).fit(X,yy)
    lag=G[:,t]; r3=G[:,t-2:t+1].mean(1)
    P={"persistence":lag,"unit_mean":CS[:,t]/(t+1),"roll3":r3,"ar1":ar.predict(Xte[:,[0]]),"linear":lin.predict(Xte),"rf":rf.predict(Xte),"gb":gb.predict(Xte)}
    for m,p in P.items(): sse[m][k]=((y-p)**2).sum(); sae[m][k]=np.abs(y-p).sum()
    sst[k]=((y-y.mean())**2).sum(); nn[k]=U
    hrs=HG[:,t]>HG[:,t-1]
    pol={"lag_threshold":lag<EV,"roll3_threshold":r3<EV,"forecast_linear":P["linear"]<EV,"forecast_gb":P["gb"]<EV,"hours_rising_only":hrs,
         "forecast_linear_AND_hours":(P["linear"]<EV)&hrs,"lag_AND_hours":(lag<EV)&hrs}
    for p,a in pol.items(): cnt[p][k]=C(a,ev)
    for th in THS:
        swp[f"lag|{th}"][k]=C(lag<th,ev); swp[f"forecast_linear|{th}"][k]=C(P["linear"]<th,ev); swp[f"forecast_linear_AND_hours|{th}"][k]=C((P["linear"]<th)&hrs,ev)
    # unseen units
    for fi,hold in enumerate(FOLDS):
        trn=np.setdiff1d(np.arange(U),hold); Xg,yg=trainset(t,trn); lg=LinearRegression().fit(Xg,yg)
        if t in RETRAIN or fi not in grf:
            grf[fi]=RandomForestRegressor(80,max_depth=5,random_state=0,n_jobs=1).fit(Xg,yg); ggb[fi]=GradientBoostingRegressor(n_estimators=80,max_depth=3,learning_rate=0.08,random_state=0).fit(Xg,yg)
        Xh=Xte[hold]; yh=y[hold]
        for m,p in (("persistence",lag[hold]),("linear",lg.predict(Xh)),("rf",grf[fi].predict(Xh)),("gb",ggb[fi].predict(Xh))): gsse[m][k]+=((yh-p)**2).sum()
        gn[k]+=len(hold)
# ---- DEA aligned to origin (levels over trailing windows ending at origin), 3 windows
DEA={}; DEA_O=[o for o in ORIG if o%3==0]
for w in (4,8,"expanding"):
    rho=[];rl=[];d2=[];d2a=[];base=[];cov=[]
    for t in DEA_O:
        hi=t+OFF+1; lo=0 if w=="expanding" else hi-w
        x=np.column_stack([LV["hours"][:,lo:hi].mean(1),LV["employment"][:,lo:hi].mean(1)]); o=LV["gdp"][:,lo:hi].mean(1)[:,None]
        x=x/x.mean(0); o=o/o.mean(0); sc=np.array([ccr(x,o,i) for i in range(U)]); ok=~np.isnan(sc); cov.append(ok.mean()); y=G[:,t+1]; lag=G[:,t]
        rho.append(spearmanr(sc[ok],y[ok])[0]); rl.append(spearmanr(lag[ok],y[ok])[0])
        def r2(A,adj=False):
            m=LinearRegression().fit(A,y[ok]); r=m.score(A,y[ok]); n_=ok.sum(); k_=A.shape[1]
            return 1-(1-r)*(n_-1)/(n_-k_-1) if adj else r
        A1=lag[ok][:,None]; A2=np.column_stack([lag[ok],sc[ok]])
        d2.append(r2(A2)-r2(A1)); d2a.append(r2(A2,True)-r2(A1,True)); base.append(r2(A1))
    DEA[str(w)]=dict(rho=np.array(rho),rho_lag=np.array(rl),dR2=np.array(d2),dR2_adj=np.array(d2a),base_R2=np.array(base),cov=float(np.mean(cov)),origins=DEA_O)
# ---- block bootstrap over origins
B=2000; rb=np.random.default_rng(5); nO=len(ORIG); BL=4
def resample():
    starts=rb.integers(0,nO,int(np.ceil(nO/BL))); idx=np.concatenate([(s+np.arange(BL))%nO for s in starts])[:nO]; return idx
IDX=[resample() for _ in range(B)]
def ci(f): 
    pt=f(np.arange(nO)); bs=np.array([f(i) for i in IDX]); return [float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
rm=lambda m,i:np.sqrt(sse[m][i].sum()/nn[i].sum())
def prf(p,i,src=cnt):
    c=src[p][i].sum(0); tp,fp,fn,tn=c; pr=tp/(tp+fp) if tp+fp else np.nan; rc=tp/(tp+fn) if tp+fn else np.nan; f1=2*tp/(2*tp+fp+fn) if tp else 0.0; return pr,rc,f1
cost=lambda p,i,w,src=cnt:(w*src[p][i,2]+src[p][i,1]).sum()
out={"units":LEAF,"U":U,"T":T,"n_origins":nO,"n_cases":int(nn.sum()),"first_target":dates[ORIG[0]+1],"last_target":dates[ORIG[-1]+1],"n_events":int(cnt["lag_threshold"][:,0].sum()+cnt["lag_threshold"][:,2].sum())}
out["event_rate"]=out["n_events"]/out["n_cases"]
out["accuracy"]={m:{"rmse":ci(lambda i,m=m:rm(m,i)),"mae":ci(lambda i,m=m:sae[m][i].sum()/nn[i].sum()),"r2":ci(lambda i,m=m:1-sse[m][i].sum()/sst[i].sum()),
   "d_vs_persistence":ci(lambda i,m=m:rm(m,i)-rm("persistence",i)),"d_vs_linear":ci(lambda i,m=m:rm(m,i)-rm("linear",i))} for m in MODELS}
out["unseen_units"]={m:{"rmse":ci(lambda i,m=m:np.sqrt(gsse[m][i].sum()/gn[i].sum())),"d_vs_persistence":ci(lambda i,m=m:np.sqrt(gsse[m][i].sum()/gn[i].sum())-np.sqrt(gsse["persistence"][i].sum()/gn[i].sum()))} for m in gsse}
out["policies"]={p:{"precision":ci(lambda i,p=p:prf(p,i)[0]),"recall":ci(lambda i,p=p:prf(p,i)[1]),"f1":ci(lambda i,p=p:prf(p,i)[2]),"n_alerts":int(cnt[p][:,0].sum()+cnt[p][:,1].sum()),
   **{f"cost_w{w}":ci(lambda i,p=p,w=w:cost(p,i,w)) for w in (1,3,10)}} for p in POL}
def pwd(a,b): return {f"w{w}":ci(lambda i,w=w:cost(a,i,w)-cost(b,i,w)) for w in (1,3,10)}
out["pairwise"]={"forecast_linear - lag_threshold":pwd("forecast_linear","lag_threshold"),"forecast_linear_AND_hours - forecast_linear":pwd("forecast_linear_AND_hours","forecast_linear"),
   "forecast_gb - forecast_linear":pwd("forecast_gb","forecast_linear"),"forecast_linear - roll3_threshold":pwd("forecast_linear","roll3_threshold")}
out["sweep"]={k:{"precision":prf(k,np.arange(nO),swp)[0],"recall":prf(k,np.arange(nO),swp)[1],**{f"cost_w{w}":float(cost(k,np.arange(nO),w,swp)) for w in (1,3,10)}} for k in swp}
def cib(a): 
    a=np.asarray(a); return None
out["dea"]={}
for w,D in DEA.items():
    n=len(D["rho"]); bsidx=[rb.integers(0,n,n) for _ in range(B)]
    f=lambda v:[float(v.mean()),float(np.percentile([v[i].mean() for i in bsidx],2.5)),float(np.percentile([v[i].mean() for i in bsidx],97.5))]
    out["dea"][w]={"rho":f(D["rho"]),"rho_lag":f(D["rho_lag"]),"dR2":f(D["dR2"]),"dR2_adj":f(D["dR2_adj"]),"base_R2":f(D["base_R2"]),"coverage":D["cov"],"n_origins":n}
# events by calendar year
yr=pd.Series({dates[t+1][:4]:0 for t in ORIG}); 
for k,t in enumerate(ORIG): yr[dates[t+1][:4]]+=int(cnt["lag_threshold"][k,0]+cnt["lag_threshold"][k,2])
out["events_by_year"]=yr.to_dict()
json.dump(out,open("real_results.json","w"),indent=1); print("done",out["n_cases"],out["n_events"])
