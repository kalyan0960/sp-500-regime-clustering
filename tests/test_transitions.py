"""Deterministic Stage 11 filtered-state transition-target tests."""
import numpy as np
import pandas as pd
import pytest
from market_regime.transitions import construct_transition_targets, filtered_transition_counts

def frame(states=(1, 1, 3, 2, 4, 4), start="2017-12-27"):
    n = len(states)
    return pd.DataFrame({"Date": pd.bdate_range(start, periods=n), "HMM_Eligible": True,
        "HMM_Filtered_State": pd.Series(states, dtype="Int64"),
        "HMM_Filtered_Low_Confidence": [i == 2 for i in range(n)],
        "Abnormal_Volume": np.arange(n), "VIX_Close": np.arange(n)+10, "Original": np.arange(n)})

@pytest.fixture
def built(): return construct_transition_targets(frame(), horizons=(1, 3))

def test_any_transition_definition(built): assert built["Any_Transition_Within_3D"].iloc[0] == 1
def test_endpoint_transition_is_distinct(built): assert built["Endpoint_Transition_3D"].iloc[0] == 1
def test_stress_entry_applicability_and_value(built):
    assert built["Stress_Entry_Within_3D"].iloc[0] == 1 and pd.isna(built["Stress_Entry_Within_3D"].iloc[2])
def test_stress_exit_applicability_and_value(built):
    assert built["Stress_Exit_Within_1D"].iloc[2] == 1 and pd.isna(built["Stress_Exit_Within_1D"].iloc[1])
def test_acute_entry_applicability_and_value(built):
    assert built["Acute_Entry_Within_3D"].iloc[1] == 1 and pd.isna(built["Acute_Entry_Within_1D"].iloc[4])
def test_worsening_and_improving_applicability(built):
    assert built["Stress_Worsening_Within_1D"].iloc[1] == 1 and pd.isna(built["Stress_Worsening_Within_1D"].iloc[4])
    assert built["Stress_Improving_Within_1D"].iloc[2] == 1 and pd.isna(built["Stress_Improving_Within_1D"].iloc[0])
def test_first_transition_lead_and_destination(built):
    assert built["First_Transition_Lead_3D"].iloc[0] == 2 and built["First_Transition_Destination_3D"].iloc[0] == 3
def test_endpoint_minimum_and_maximum(built):
    assert tuple(built.loc[0,["State_At_Horizon_3D","Minimum_State_Within_3D","Maximum_State_Within_3D"]]) == (2,1,3)
def test_adjacent_change_count(built): assert built["Number_of_State_Changes_Within_3D"].iloc[0] == 2
def test_complete_window_requirement():
    source=frame(); source.loc[2,"HMM_Eligible"]=False
    assert pd.isna(construct_transition_targets(source,horizons=(3,))["Any_Transition_Within_3D"].iloc[0])
def test_last_h_rows_unavailable(built): assert built["Any_Transition_Within_3D"].tail(3).isna().all()
def test_ineligible_current_row_is_unavailable():
    source=frame(); source.loc[0,"HMM_Eligible"]=False
    assert construct_transition_targets(source,horizons=(1,)).loc[0,"Transition_Sample_1D"]=="Unavailable"
def test_structural_missing_survives_csv_reload(tmp_path):
    out=construct_transition_targets(frame(),horizons=(1,)); path=tmp_path/"x.csv"; out.to_csv(path,index=False)
    assert pd.isna(pd.read_csv(path)["Stress_Entry_Within_1D"].iloc[2])
def test_boundary_crossing_training_rows_are_purged(built): assert "Purged" in set(built["Transition_Sample_3D"])
def test_post_cutoff_complete_windows_are_test():
    out=construct_transition_targets(frame(start="2018-01-02"),horizons=(1,)); assert set(out.Transition_Sample_1D)=={"Test","Unavailable"}
def test_horizon_uses_observations_not_calendar_days():
    source=frame(states=(1,3,3),start="2018-01-05"); out=construct_transition_targets(source,horizons=(1,))
    assert (source.Date.iloc[1]-source.Date.iloc[0]).days==3 and out.Any_Transition_Within_1D.iloc[0]==1
def test_confidence_flags_use_saved_threshold_flags(built):
    assert built.Future_Window_Contains_Low_Confidence_3D.iloc[0] and not built.Transition_Window_All_High_Confidence_3D.iloc[0]
def test_volume_and_vix_do_not_change_targets():
    a=frame(); b=a.copy(); b.Abnormal_Volume=1e9; b.VIX_Close=-1e9
    x=construct_transition_targets(a,horizons=(3,)); y=construct_transition_targets(b,horizons=(3,))
    columns=[c for c in x if c not in a.columns]
    pd.testing.assert_frame_equal(x[columns],y[columns])
def test_future_change_affects_only_containing_windows():
    a=frame(states=(1,1,1,1,1,1)); b=a.copy(); b.loc[4,"HMM_Filtered_State"]=3
    x=construct_transition_targets(a,horizons=(3,)); y=construct_transition_targets(b,horizons=(3,))
    assert x.Any_Transition_Within_3D.iloc[0]==y.Any_Transition_Within_3D.iloc[0] and y.Any_Transition_Within_3D.iloc[1]==1
def test_input_columns_and_row_count_preserved(built):
    source=frame(); assert len(built)==len(source); pd.testing.assert_frame_equal(built[source.columns],source)
def test_dates_remain_unique_and_chronological(built): assert built.Date.is_unique and built.Date.is_monotonic_increasing
def test_transition_counts_are_row_normalized():
    counts=filtered_transition_counts(frame(start="2018-01-02")); sums=counts[counts.Sample.eq("All")].groupby("From_State").Empirical_Probability.sum()
    assert np.allclose(sums,1)
def test_repeated_runs_are_deterministic():
    pd.testing.assert_frame_equal(construct_transition_targets(frame(),horizons=(1,3)),construct_transition_targets(frame(),horizons=(1,3)))
