"""Leakage-safe Stage 13 logistic transition prediction utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
 log_loss, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score)

SEED=13013; C_GRID=(.01,.1,1.,10.); STATES=(1,2,3,4)
BASE_CONT=("Log_Return","GARCH_Volatility_TrainFit","Drawdown_252","VIX_Close")
VOLUME="Log_Abnormal_Volume"; PRIMARY="Any_Transition_Within_5D"

def prepare(frame,outcome=PRIMARY,horizon=5,model="Model 2"):
    sample=f"Transition_Sample_{horizon}D"; cols=["Date",sample,outcome,"HMM_Filtered_State",*BASE_CONT]
    if model in ("Model 2","Model 3"): cols += [VOLUME]
    d=frame.loc[frame[sample].isin(["Train","Test"]),cols].copy().replace([np.inf,-np.inf],np.nan).dropna()
    d["Date"]=pd.to_datetime(d.Date)
    if model=="Model 3":
        d["Volatility_Category"]=pd.cut(d.GARCH_Volatility_TrainFit,[-np.inf,.007142284694485,.010788124806444099,np.inf],labels=["Low","Medium","High"])
        d["LogVol_x_Medium"]=d[VOLUME]*(d.Volatility_Category=="Medium")
        d["LogVol_x_High"]=d[VOLUME]*(d.Volatility_Category=="High")
    return d.sort_values("Date").reset_index(drop=True)

def expanding_folds(n,horizon=5,n_splits=5):
    starts=np.linspace(int(n*.4),int(n*.8),n_splits,dtype=int); ends=np.r_[starts[1:],n]
    return [(np.arange(s-horizon),np.arange(s,e)) for s,e in zip(starts,ends)]

def pipeline(model="Model 2",C=1.):
    cont=[] if model=="Model 0" else list(BASE_CONT)+([VOLUME] if model in ("Model 2","Model 3") else [])+(["LogVol_x_Medium","LogVol_x_High"] if model=="Model 3" else [])
    transforms=[("state",OneHotEncoder(categories=[list(STATES)],drop=[1],handle_unknown="error",sparse_output=False),["HMM_Filtered_State"]),("continuous",StandardScaler(),cont)]
    if model=="Model 3": transforms.append(("volatility_category",OneHotEncoder(categories=[["Low","Medium","High"]],drop=["Low"],handle_unknown="error",sparse_output=False),["Volatility_Category"]))
    pre=ColumnTransformer(transforms,verbose_feature_names_out=False)
    return Pipeline([("preprocess",pre),("logistic",LogisticRegression(C=C,max_iter=5000,random_state=SEED))])

def tune_oof(train,model,horizon=5):
    y=train.iloc[:,2].astype(int); folds=expanding_folds(len(train),horizon); rows=[]; scores={}
    for C in C_GRID:
      losses=[]
      for k,(ti,vi) in enumerate(folds,1):
        fit=pipeline(model,C).fit(train.iloc[ti],y.iloc[ti]); pr=fit.predict_proba(train.iloc[vi])[:,1]; ll=log_loss(y.iloc[vi],pr); losses.append(ll)
        rows.append(dict(Model=model,Fold=k,Train_Start=train.Date.iloc[ti[0]],Train_End=train.Date.iloc[ti[-1]],Validation_Start=train.Date.iloc[vi[0]],Validation_End=train.Date.iloc[vi[-1]],Purge_Size=horizon,Train_N=len(ti),Validation_N=len(vi),Train_Events=int(y.iloc[ti].sum()),Validation_Events=int(y.iloc[vi].sum()),C=C,Log_Loss=ll,ROC_AUC=roc_auc_score(y.iloc[vi],pr),PR_AUC=average_precision_score(y.iloc[vi],pr)))
      scores[C]=np.mean(losses)
    best=min(scores,key=scores.get); oof=np.full(len(train),np.nan)
    for ti,vi in folds: oof[vi]=pipeline(model,best).fit(train.iloc[ti],y.iloc[ti]).predict_proba(train.iloc[vi])[:,1]
    ok=np.isfinite(oof); candidates=np.unique(oof[ok]); threshold=float(candidates[np.argmax([f1_score(y[ok],oof[ok]>=x) for x in candidates])])
    table=pd.DataFrame(rows); table["Selected_C"]=table.C.eq(best); return best,threshold,oof,table

def metrics(y,p,threshold=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=p>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    bins=np.clip(np.digitize(p,np.linspace(0,1,11))-1,0,9); ece=sum(np.mean(bins==b)*abs(y[bins==b].mean()-p[bins==b].mean()) for b in range(10) if np.any(bins==b))
    import statsmodels.api as sm
    z=np.log(np.clip(p,1e-8,1-1e-8)/(1-np.clip(p,1e-8,1-1e-8)))
    if np.ptp(z)==0: cal_intercept=float(np.log(y.mean()/(1-y.mean()))); cal_slope=np.nan
    else:
        cal=sm.Logit(y,sm.add_constant(z)).fit(disp=0); cal_intercept,cal_slope=cal.params
    return dict(ROC_AUC=roc_auc_score(y,p),PR_AUC=average_precision_score(y,p),Brier=brier_score_loss(y,p),Log_Loss=log_loss(y,p),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred),F1=f1_score(y,pred),Specificity=tn/(tn+fp),Balanced_Accuracy=balanced_accuracy_score(y,pred),TN=tn,FP=fp,FN=fn,TP=tp,Calibration_Intercept=cal_intercept,Calibration_Slope=cal_slope,ECE_10_Bins=ece,Observed_Rate=y.mean(),Predicted_Rate=p.mean())

def paired_bootstrap(y,p1,p2,reps=2000,block=20,seed=SEED):
    rng=np.random.default_rng(seed); n=len(y); vals=[]; failed=0; keys=("ROC_AUC","PR_AUC","Brier","Log_Loss")
    for _ in range(reps):
      ix=np.concatenate([np.arange(s,s+block) for s in rng.integers(0,n-block+1,int(np.ceil(n/block)))])[:n]
      if np.unique(np.asarray(y)[ix]).size<2: failed+=1; continue
      yy=np.asarray(y)[ix]; a=np.asarray(p1)[ix]; b=np.asarray(p2)[ix]
      vals.append([roc_auc_score(yy,b)-roc_auc_score(yy,a),average_precision_score(yy,b)-average_precision_score(yy,a),brier_score_loss(yy,b)-brier_score_loss(yy,a),log_loss(yy,b)-log_loss(yy,a)])
    arr=np.asarray(vals); return pd.DataFrame({"Metric":keys,"CI_Lower":np.quantile(arr,.025,axis=0),"CI_Upper":np.quantile(arr,.975,axis=0),"Valid_Replications":len(arr),"Failed_Replications":failed})
