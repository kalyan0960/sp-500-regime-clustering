import numpy as np,pandas as pd
from market_regime.clustering import FEATURES,eligible,fit,evaluate
def test_features_and_leakage():
 assert FEATURES==['Log_Return','GARCH_Volatility_TrainFit','Drawdown_252']
 i=pd.date_range('2017-01-01',periods=30); f=pd.DataFrame({'Log_Return':np.arange(30.),'GARCH_Volatility_TrainFit':1.,'Drawdown_252':-1.,'Abnormal_Volume':9.,'VIX_Close':8.},index=i); mask,tr,te,s,x,_=fit(f,'2017-01-20'); f2=f.copy(); f2.loc[f2.index>='2017-01-21','Log_Return']=999; _,_,_,s2,x2,_=fit(f2,'2017-01-20'); assert np.allclose(s.mean_,s2.mean_) and np.allclose(x,x2) and eligible(f).all()
