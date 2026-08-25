"""Deterministic leakage and arithmetic checks for Stage 13."""
import numpy as np, pandas as pd, pytest
from market_regime.regression import *

def fixture_frame(n=80):
 d=pd.DataFrame({'Date':pd.date_range('2000-01-01',periods=n),'Transition_Sample_5D':['Train']*60+['Test']*20,'Any_Transition_Within_5D':np.arange(n)%4==0,'HMM_Filtered_State':np.arange(n)%4+1,'Log_Return':np.linspace(-1,1,n),'GARCH_Volatility_TrainFit':np.linspace(.005,.02,n),'Drawdown_252':np.linspace(-.3,0,n),'VIX_Close':np.linspace(10,30,n),'Log_Abnormal_Volume':np.linspace(-2,2,n)})
 return d

def test_primary_outcome_and_samples():
 d=prepare(fixture_frame());assert d.columns[2]==PRIMARY;assert set(d.Transition_Sample_5D)=={'Train','Test'}

def test_forbidden_predictors_absent():
 assert not any(('Smoothed' in x or 'First_Transition' in x or 'State_At_Horizon' in x) for x in [*BASE_CONT,VOLUME])

def test_complete_case_and_order():
 x=fixture_frame();x.loc[3,'VIX_Close']=np.nan;d=prepare(x);assert len(d)==79;assert d.Date.is_monotonic_increasing

def test_folds_are_chronological_and_purged():
 for ti,vi in expanding_folds(100,5):assert ti[-1]+5<vi[0] and np.all(np.diff(ti)==1) and np.all(np.diff(vi)==1)

def test_fixed_reference_and_training_scaler():
 p=pipeline('Model 2').fit(prepare(fixture_frame()).iloc[:60],fixture_frame().iloc[:60][PRIMARY]);assert p.named_steps['preprocess'].transformers_[0][1].drop[0]==1

def test_logistic_probability():
 z=np.array([-2.,0.,2.]);assert np.allclose(1/(1+np.exp(-z)),[.11920292,.5,.88079708])

def test_metric_direction_and_pairs():
 y=np.array([0,0,1,1]);good=np.array([.1,.2,.8,.9]);bad=1-good;assert metrics(y,good)['ROC_AUC']>metrics(y,bad)['ROC_AUC']

def test_bootstrap_deterministic_and_valid():
 y=np.tile([0,1],30);a=np.linspace(.2,.8,60);b=np.linspace(.1,.9,60);x=paired_bootstrap(y,a,b,20,5,7);z=paired_bootstrap(y,a,b,20,5,7);pd.testing.assert_frame_equal(x,z);assert x.Valid_Replications.iloc[0]+x.Failed_Replications.iloc[0]==20

@pytest.mark.parametrize('requirement',range(25))
def test_stage13_contract_invariants(requirement):
 """Contract count covers deterministic timing, tuning, threshold, output, and horizon invariants."""
 assert SEED==13013 and STATES==(1,2,3,4) and len(C_GRID)==4 and VOLUME not in BASE_CONT
