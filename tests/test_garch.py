import numpy as np, pandas as pd
from market_regime.garch import prepare_percent_returns, recursive_variance
def test_scaling_and_recursion_no_lookahead():
    r=pd.Series([.01,.02,.03],index=pd.date_range('2020-01-01',periods=3)); p=prepare_percent_returns(r); assert p.iloc[0]==1
    v=recursive_variance(p,mu=0,omega=.1,alpha=.2,beta=.7,initial_variance=1); assert v.iloc[0]==1; assert v.iloc[1]==.1+.2*1**2+.7
    extended=pd.concat([p,pd.Series([99.],index=[pd.Timestamp('2020-01-04')])]); assert np.allclose(v,recursive_variance(extended,mu=0,omega=.1,alpha=.2,beta=.7,initial_variance=1).iloc[:3])
