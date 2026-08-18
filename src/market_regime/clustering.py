"""Training-only revised K-means baseline utilities."""
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score,adjusted_rand_score
FEATURES=['Log_Return','GARCH_Volatility_TrainFit','Drawdown_252']
def eligible(frame): return np.isfinite(frame[FEATURES]).all(axis=1)
def fit(frame,training_end,seed=42):
    f=frame.copy(); mask=eligible(f); tr=mask&(f.index<=training_end); te=mask&(f.index>training_end); sc=StandardScaler().fit(f.loc[tr,FEATURES]); return mask,tr,te,sc,sc.transform(f.loc[tr,FEATURES]),sc.transform(f.loc[te,FEATURES])
def evaluate(Xtr,Xte,k,seed=42):
    m=KMeans(k,random_state=seed,n_init=20).fit(Xtr); a=m.predict(Xte); d=np.linalg.norm(Xte-m.cluster_centers_[a],axis=1); return m,a,d,{'K':k,'inertia':m.inertia_,'silhouette':silhouette_score(Xtr,m.labels_),'davies_bouldin':davies_bouldin_score(Xtr,m.labels_),'calinski_harabasz':calinski_harabasz_score(Xtr,m.labels_),'iterations':m.n_iter_}
def stability(X,k):
    labels=[KMeans(k,random_state=s,n_init=20).fit_predict(X) for s in range(20)]; ar=[adjusted_rand_score(labels[0],x) for x in labels[1:]]; return {'K':k,'ari_mean':np.mean(ar),'ari_min':np.min(ar),'ari_std':np.std(ar)}
