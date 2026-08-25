"""Deterministic safeguards for Stage 12 volume association analysis."""
import numpy as np
import pandas as pd
import pytest
from market_regime.volume_analysis import *

def data(test_extreme=False):
    n=36; dates=pd.bdate_range("2017-11-15",periods=n); transition=np.tile([0,0,1],12)
    frame=pd.DataFrame({"Date":dates,"Transition_Sample_5D":["Train"]*30+["Test"]*6,
        PRIMARY_TARGET:pd.Series(transition,dtype="Int64"),OUTCOME:.1*transition+np.linspace(-.2,.2,n),
        "Abnormal_Volume":np.exp(.1*transition+np.linspace(-.2,.2,n)),VOLATILITY:np.arange(n,dtype=float),
        "Transition_Window_All_High_Confidence_5D":True,"HMM_Filtered_State":np.tile([1,1,2,2,3,4],6),
        "HMM_Smoothed_State":4,"VIX_Close":20.,"Original":range(n)})
    for t in SECONDARY_TARGETS: frame[t]=frame[PRIMARY_TARGET]
    if test_extreme: frame.loc[30:,OUTCOME]=1e6; frame.loc[30:,VOLATILITY]=1e6
    return frame

def test_training_only_tertiles(): assert training_tertiles(primary_sample(data()))==pytest.approx((29/3,58/3))
def test_category_assignment(): assert list(assign_volatility_category(pd.Series([0,10,30]),(9,20)).astype(str))==["Low","Medium","High"]
def test_test_values_do_not_change_thresholds(): assert prepare_primary(data())[1]==prepare_primary(data(True))[1]
def test_primary_restriction(): assert len(primary_sample(data()))==30
def test_structural_missing_excluded():
    x=data(); x.loc[2,PRIMARY_TARGET]=pd.NA; assert len(primary_sample(x))==29
def test_design_terms_present():
    s,_=prepare_primary(data()); a,_,_=fit_two_factor(s); assert a.Term.str.contains("Transition").any() and a.Term.str.contains("Volatility").any()
def test_reference_categories_documented():
    s,_=prepare_primary(data()); _,h,_=fit_two_factor(s); assert h.Reference_Categories.str.contains("Transition=0; Volatility=Low").all()
def test_hac_lag_is_five():
    s,_=prepare_primary(data()); _,h,_=fit_two_factor(s); assert set(h.HAC_Max_Lag)=={5}
def test_moving_blocks_are_chronological_within_blocks():
    idx=moving_block_indices(30,5,np.random.default_rng(1)); assert all(np.all(np.diff(chunk)==1) for chunk in idx.reshape(-1,5))
def test_bootstrap_reproducible():
    s,_=prepare_primary(data()); pd.testing.assert_frame_equal(block_bootstrap(s,20,5,2),block_bootstrap(s,20,5,2))
def test_bootstrap_reports_valid_and_failed():
    s,_=prepare_primary(data()); b=block_bootstrap(s,20,5,2); assert {"Valid_Replications","Failed_Replications"}.issubset(b)
def test_mean_difference_direction():
    s,_=prepare_primary(data()); assert primary_effects(s).iloc[0].Mean_Difference==pytest.approx(s[s.Transition==1][OUTCOME].mean()-s[s.Transition==0][OUTCOME].mean())
def test_hedges_g_direction(): assert hedges_g([2,3,4],[0,1,2])>0
def test_rank_effect_direction(): assert cliff_delta([3,4],[0,1])>0
def test_inference_methods_distinguished():
    s,_=prepare_primary(data()); a,_,_=fit_two_factor(s); assert set(a.Inference_Method)=={"Classical Type III ANOVA","HC3 robust Type III"}
def test_holm_adjustment_available():
    from statsmodels.stats.multitest import multipletests; raw=np.array([.01,.02,.5]); assert np.all(multipletests(raw,method="holm")[1]>=raw)
def test_unique_event_dates():
    w=event_windows(data(),radius=2); assert w.groupby("Event_ID").Transition_Date.nunique().eq(1).all()
def test_origin_destination_correct():
    x=data(); w=event_windows(x,radius=1); first=w[w.Event_ID==1].iloc[0]; i=x.index[pd.to_datetime(x.Date)==pd.Timestamp(first.Transition_Date)][0]; assert first.Origin_State==x.HMM_Filtered_State.iloc[i-1]
def test_event_time_indexing(): assert set(event_windows(data(),radius=2).Event_Time)==set(range(-2,3))
def test_post_event_not_pre_event(): assert event_windows(data(),radius=2).query("Event_Time>0").Event_Time.gt(0).all()
def test_overlaps_flagged(): assert event_windows(data(),radius=10).Another_Transition_In_Window.any()
def test_no_smoothed_state_used():
    a=data(); b=a.copy(); b.HMM_Smoothed_State=1; pd.testing.assert_frame_equal(primary_sample(a),primary_sample(b).assign(HMM_Smoothed_State=4))
def test_no_test_period_in_primary(): assert primary_sample(data()).Transition_Sample_5D.eq("Train").all()
def test_original_columns_unchanged():
    x=data(); y=primary_sample(x); pd.testing.assert_frame_equal(y[x.columns],x.iloc[:30])
def test_row_order_and_dates_unchanged(): assert primary_sample(data()).Date.is_monotonic_increasing
def test_deterministic_analysis_outputs():
    a,_=prepare_primary(data()); pd.testing.assert_frame_equal(primary_effects(a),primary_effects(a))
def test_test_volume_does_not_change_training_inference():
    a=data(); b=a.copy(); b.loc[30:,OUTCOME]=1e9; x,_=prepare_primary(a); y,_=prepare_primary(b); pd.testing.assert_frame_equal(primary_effects(x),primary_effects(y))
def test_vix_does_not_change_results():
    a=data(); b=a.copy(); b.VIX_Close=1e9; x,_=prepare_primary(a); y,_=prepare_primary(b); pd.testing.assert_frame_equal(primary_effects(x),primary_effects(y))
