"""Annual state-level check on DOSM 'Annual productivity by state' (output per employee, CC BY 4.0). Usage: python run_state_check.py state_raw.csv -> state_results.json
Units: state x sector (p1-p5) with complete series (16 states; Supra and Malaysia excluded as units; Malaysia used as national feature). Growth years 2016-2025.
Origins t=3..8 (targets 2020-2025). Features: g(t-1), g(t-2), national same-sector g(t-1), state-total g(t-1), national-total g(t-1), unit expanding mean. Event: next-year growth < 0.
Intervals: bootstrap over STATES (clusters), B=2000. With 6 origins and one common shock the intervals do not reflect time-series uncertainty."""
import sys, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings("ignore")
q=pd.read_csv(sys.argv[1]); g=q[q.series=="growth_yoy"]; yrs=sorted(g.date.unique()); T=len(yrs)
P=g.pivot_table(index=["state","sector"],columns="date",values="output_employment")
units=[(s,sec) for (s,sec) in P.index if s not in("Supra","Malaysia") and sec!="p0" and P.loc[(s,sec)].notna().all()]
U=len(units); G=np.array([P.loc[u].to_numpy() for u in units]); states=sorted({u[0] for u in units}); sidx=np.array([states.index(u[0]) for u in units])
NS=np.array([P.loc[("Malaysia",u[1])].to_numpy() for u in units]); ST=np.array([P.loc[(u[0],"p0")].to_numpy() for u in units]); NT=P.loc[("Malaysia","p0")].to_numpy()[None,:].repeat(U,0)
CS=np.cumsum(G,1)
def feats(s): return np.column_stack([G[:,s-1],G[:,s-2],NS[:,s-1],ST[:,s-1],NT[:,s-1],CS[:,s-1]/s])
def trainset(t,m=None):
    X=np.vstack([feats(s) for s in range(2,t+1)]); y=np.concatenate([G[:,s] for s in range(2,t+1)])
    if m is not None: k=np.tile(m,t-1); X,y=X[k],y[k]
    return X,y
START=int(sys.argv[2]) if len(sys.argv)>2 else 3; ORIG=list(range(START,T-1)); MODELS=["persistence","unit_mean","ar1","linear","rf","gb"]
rec=[]; recg=[]; pol={}
FOLDS=np.array_split(np.random.default_rng(3).permutation(len(states)),4)
for t in ORIG:
    s=t+1; y=G[:,s]; X,yy=trainset(t); Xte=feats(s)
    lin=LinearRegression().fit(X,yy); ar=LinearRegression().fit(X[:,[0]],yy)
    rf=RandomForestRegressor(200,max_depth=4,random_state=0).fit(X,yy); gb=GradientBoostingRegressor(n_estimators=80,max_depth=2,learning_rate=0.05,random_state=0).fit(X,yy)
    pr={"persistence":G[:,t],"unit_mean":CS[:,t]/(t+1),"ar1":ar.predict(Xte[:,[0]]),"linear":lin.predict(Xte),"rf":rf.predict(Xte),"gb":gb.predict(Xte)}
    rec.append({"t":t,"y":y,"p":pr})
    # unseen states
    pg={m:np.zeros(U) for m in ("persistence","linear","rf","gb")}
    for f in FOLDS:
        hold=np.isin(sidx,f); Xg,yg=trainset(t,~hold); l=LinearRegression().fit(Xg,yg); r=RandomForestRegressor(200,max_depth=4,random_state=0).fit(Xg,yg); b=GradientBoostingRegressor(n_estimators=80,max_depth=2,learning_rate=0.05,random_state=0).fit(Xg,yg)
        pg["persistence"][hold]=G[hold,t]; pg["linear"][hold]=l.predict(Xte[hold]); pg["rf"][hold]=r.predict(Xte[hold]); pg["gb"][hold]=b.predict(Xte[hold])
    recg.append(pg)
B=2000; rb=np.random.default_rng(5); NSt=len(states); IDX=[rb.integers(0,NSt,NSt) for _ in range(B)]
def mask(ix):  # units belonging to resampled states (with multiplicity)
    return np.concatenate([np.where(sidx==k)[0] for k in ix])
def ci(f): 
    pt=f(np.arange(NSt)); bs=[f(ix) for ix in IDX]; return [float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
def rmse(m,ix,src="p"):
    u=mask(ix); return float(np.sqrt(np.mean(np.concatenate([(r["y"][u]-(r["p"][m][u] if src=="p" else recg[i][m][u]))**2 for i,r in enumerate(rec)]))))
def cnt(a,e): return np.array([(a&e).sum(),(a&~e).sum(),(~a&e).sum()])   # tp, fp, fn
def polc(name,ix):
    u=mask(ix); c=np.zeros(3)
    for r in rec:
        y=r["y"][u]; e=y<0
        a={"lag":G[u,r["t"]]<0,"forecast_linear":r["p"]["linear"][u]<0,"forecast_gb":r["p"]["gb"][u]<0,"forecast_rf":r["p"]["rf"][u]<0}[name]; c+=cnt(a,e)
    return c
def prf(c): tp,fp,fn=c; p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan; return p,r,(2*tp/(2*tp+fp+fn) if tp else 0.0)
out={"n_units":U,"n_states":NSt,"origins":[yrs[t+1][:4] for t in ORIG],"n_cases":U*len(ORIG)}
out["n_events"]=int(sum((r["y"]<0).sum() for r in rec)); out["event_rate"]=out["n_events"]/out["n_cases"]
out["accuracy"]={m:{"rmse":ci(lambda ix,m=m:rmse(m,ix)),"d_vs_persistence":ci(lambda ix,m=m:rmse(m,ix)-rmse("persistence",ix)),"d_vs_linear":ci(lambda ix,m=m:rmse(m,ix)-rmse("linear",ix))} for m in MODELS}
out["unseen_states"]={m:{"rmse":ci(lambda ix,m=m:rmse(m,ix,"g")),"d_vs_persistence":ci(lambda ix,m=m:rmse(m,ix,"g")-rmse("persistence",ix,"g"))} for m in ("persistence","linear","rf","gb")}
out["policies"]={n:{"precision":ci(lambda ix,n=n:prf(polc(n,ix))[0]),"recall":ci(lambda ix,n=n:prf(polc(n,ix))[1]),"f1":ci(lambda ix,n=n:prf(polc(n,ix))[2]),"cost3":ci(lambda ix,n=n:3*polc(n,ix)[2]+polc(n,ix)[1])} for n in ("lag","forecast_linear","forecast_gb")}
out["pairwise_cost3_forecast_linear_minus_lag"]=ci(lambda ix:(3*polc("forecast_linear",ix)[2]+polc("forecast_linear",ix)[1])-(3*polc("lag",ix)[2]+polc("lag",ix)[1]))
out["events_by_year"]={yrs[r["t"]+1][:4]:int((r["y"]<0).sum()) for r in rec}
json.dump(out,open(f"state_results_start{START}.json","w"),indent=1)
f=lambda x:f"{x[0]:+.2f} [{x[1]:+.2f},{x[2]:+.2f}]"
print("units",U,"states",NSt,"cases",out["n_cases"],"events",out["n_events"],round(out["event_rate"],3)); print(out["events_by_year"])
for m in MODELS: print(f"{m:12s} RMSE {out['accuracy'][m]['rmse'][0]:.2f} | vs persistence {f(out['accuracy'][m]['d_vs_persistence'])} | vs linear {f(out['accuracy'][m]['d_vs_linear'])}")
for m in out["unseen_states"]: print("unseen",m,round(out["unseen_states"][m]["rmse"][0],2),f(out["unseen_states"][m]["d_vs_persistence"]))
for n,v in out["policies"].items(): print(n,"P %.2f R %.2f F1 %.2f cost3 %.0f"%(v["precision"][0],v["recall"][0],v["f1"][0],v["cost3"][0]))
print("forecast_linear - lag cost3",f(out["pairwise_cost3_forecast_linear_minus_lag"]))
