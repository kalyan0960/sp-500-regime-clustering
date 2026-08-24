"""Deterministic unit tests for Stage 10A HMM utility safeguards."""

import numpy as np
import pandas as pd
import pytest
import market_regime.hmm_model as hmm_model

from market_regime.hmm_model import (
    FEATURES,
    HMMValidationError,
    bic,
    eligible_sample,
    episodes,
    expected_durations,
    filtered_summary,
    forward_filter,
    fit_hmm_candidate,
    fit_multiple_seeds,
    parameter_count,
    candidate_stability,
    covariance_warnings,
    centroid_distances,
    degeneracy_warnings,
    finalist_duration_summary,
    original_scale_profiles,
    ordered_stress_states,
    aligned_state_crosstab,
    reconstruct_candidate,
    ordered_state_mapping,
    reorder_probabilities,
    selected_assignments,
    training_feature_scaler,
    validate_features,
    validate_markov,
)


@pytest.fixture
def fixed_hmm():
    return {
        "start": np.array([0.5, 0.5]),
        "transition": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "means": np.array([[0.0, 0.0], [2.0, 2.0]]),
        "diag_covariances": np.array([[1.0, 1.0], [1.0, 1.0]]),
        "observations": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    }


def test_authorized_feature_list_is_exact():
    assert validate_features(FEATURES) == FEATURES


@pytest.mark.parametrize(
    "features",
    [
        ["Log_Return", "Drawdown_252", "GARCH_Volatility_TrainFit"],
        ["Log_Return", "GARCH_Volatility_FullSample", "Drawdown_252"],
        ["Log_Return", "Abnormal_Volume", "Drawdown_252"],
        ["Log_Return", "VIX_Close", "Drawdown_252"],
    ],
)
def test_unauthorized_feature_lists_are_rejected(features):
    with pytest.raises(HMMValidationError):
        validate_features(features)


def test_eligibility_filters_non_finite_rows_without_mutating_input():
    frame = pd.DataFrame(
        {
            "Log_Return": [0.01, np.nan, 0.02],
            "GARCH_Volatility_TrainFit": [0.02, 0.02, np.inf],
            "Drawdown_252": [0.0, -0.1, -0.2],
            "Auxiliary": [1, 2, 3],
        }
    )
    original = frame.copy(deep=True)
    result = eligible_sample(frame)
    assert result.index.tolist() == [0]
    assert result.columns.tolist() == frame.columns.tolist()
    pd.testing.assert_frame_equal(frame, original)


def test_parameter_counts_for_diagonal_and_full_covariance():
    assert parameter_count(3, 3, "diag") == 26
    assert parameter_count(3, 3, "full") == 35


def test_bic_uses_requested_formula():
    assert bic(log_likelihood=-10.0, parameter_count_value=2, n=100) == pytest.approx(
        20.0 + 2 * np.log(100)
    )


def test_markov_validation_requires_row_stochastic_transition_matrix():
    start, transition = validate_markov([0.4, 0.6], [[0.8, 0.2], [0.1, 0.9]])
    assert np.isclose(start.sum(), 1.0)
    assert np.allclose(transition.sum(axis=1), 1.0)
    with pytest.raises(HMMValidationError):
        validate_markov([0.4, 0.6], [[0.8, 0.3], [0.1, 0.9]])


def test_expected_duration_uses_diagonal_transition_probability():
    durations = expected_durations(np.array([[0.8, 0.2], [0.1, 0.9]]))
    assert np.allclose(durations, [5.0, 10.0])


def test_episode_extraction_returns_empirical_durations():
    assert episodes([0, 0, 1, 1, 1, 0]) == [
        {"state": 0, "start_index": 0, "end_index": 1, "duration": 2},
        {"state": 1, "start_index": 2, "end_index": 4, "duration": 3},
        {"state": 0, "start_index": 5, "end_index": 5, "duration": 1},
    ]


def test_forward_filter_is_normalized_and_reproducible(fixed_hmm):
    probabilities = forward_filter(
        fixed_hmm["observations"],
        fixed_hmm["start"],
        fixed_hmm["transition"],
        fixed_hmm["means"],
        fixed_hmm["diag_covariances"],
    )
    repeated = forward_filter(
        fixed_hmm["observations"],
        fixed_hmm["start"],
        fixed_hmm["transition"],
        fixed_hmm["means"],
        fixed_hmm["diag_covariances"],
    )
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.allclose(probabilities, repeated)


def test_forward_filter_uses_no_future_observations(fixed_hmm):
    filtered = forward_filter(
        fixed_hmm["observations"],
        fixed_hmm["start"],
        fixed_hmm["transition"],
        fixed_hmm["means"],
        fixed_hmm["diag_covariances"],
    )
    extended = forward_filter(
        np.vstack([fixed_hmm["observations"], [99.0, 99.0]]),
        fixed_hmm["start"],
        fixed_hmm["transition"],
        fixed_hmm["means"],
        fixed_hmm["diag_covariances"],
    )
    assert np.allclose(filtered, extended[: len(filtered)])


def test_forward_filter_supports_full_covariance(fixed_hmm):
    full_covariances = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    probabilities = forward_filter(
        fixed_hmm["observations"],
        fixed_hmm["start"],
        fixed_hmm["transition"],
        fixed_hmm["means"],
        full_covariances,
        covariance_type="full",
    )
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_low_confidence_is_strictly_below_threshold():
    summary = filtered_summary([[0.60, 0.40], [0.59, 0.41]])
    assert summary["HMM_Low_Confidence"].tolist() == [False, True]


def test_training_scaling_excludes_later_observations():
    frame = pd.DataFrame(
        {
            "Date": ["2017-12-29", "2017-12-31", "2018-01-02"],
            "Log_Return": [0.0, 2.0, 1_000.0],
            "GARCH_Volatility_TrainFit": [1.0, 3.0, 1_000.0],
            "Drawdown_252": [-0.1, -0.3, 1_000.0],
        }
    )
    scaler, training, scaled = training_feature_scaler(frame, "2017-12-31")
    assert len(training) == 2
    assert np.allclose(scaler.mean_, [1.0, 2.0, -0.2])
    assert np.allclose(scaled.mean(axis=0), 0.0)


def test_candidate_fitting_is_reproducible(fixed_hmm):
    observations = np.vstack([np.zeros((20, 2)), np.full((20, 2), 3.0)])
    first = fit_hmm_candidate(observations, 2, "diag", seed=7, n_iter=50)
    second = fit_hmm_candidate(observations, 2, "diag", seed=7, n_iter=50)
    assert first["converged"] and second["converged"]
    assert first["log_likelihood"] == pytest.approx(second["log_likelihood"])
    assert np.array_equal(first["states"], second["states"])


def test_best_run_selection_retains_highest_converged_likelihood(monkeypatch):
    def fake_fit(_observations, _k, _covariance, seed, **_kwargs):
        return {"seed": seed, "converged": seed != 2, "log_likelihood": float(seed), "status": "converged" if seed != 2 else "failed"}

    monkeypatch.setattr(hmm_model, "fit_hmm_candidate", fake_fit)
    result = fit_multiple_seeds(np.ones((2, 1)), 2, "diag", [1, 2, 3])
    assert result["best_run"]["seed"] == 3


def test_failed_fit_is_recorded():
    class FailingModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _observations):
            raise RuntimeError("deterministic fit failure")

    result = fit_hmm_candidate(np.ones((5, 1)), 2, "diag", seed=1, model_factory=FailingModel)
    assert result["status"] == "failed"
    assert "RuntimeError" in result["error"]


def test_stability_and_degeneracy_detection():
    runs = [
        {"converged": True, "states": np.array([0, 0, 1, 1])},
        {"converged": True, "states": np.array([1, 1, 0, 0])},
    ]
    assert candidate_stability(runs)["ari_mean"] == pytest.approx(1.0)
    warning = degeneracy_warnings(np.array([0, 0, 0]), 2)
    assert warning["degenerate_states"] == [1]


def test_covariance_validation_flags_near_singularity():
    class DummyModel:
        covariance_type = "diag"
        covars_ = np.array(
            [[[1.0, 0.0], [0.0, 1e-8]], [[2.0, 0.0], [0.0, 3.0]]]
        )

    warning = covariance_warnings(DummyModel())
    assert warning["near_singular_states"] == [0]


def test_finalist_reconstruction_profiles_durations_and_distances():
    observations = np.vstack([np.zeros((10, 2)), np.full((10, 2), 3.0)])
    fitted = fit_hmm_candidate(observations, 2, "diag", seed=7, n_iter=50)
    parameters = {"seed": 7, "startprob": fitted["model"].startprob_.tolist(),
                  "transmat": fitted["model"].transmat_.tolist(), "means": fitted["model"].means_.tolist(),
                  "covars": np.asarray(fitted["model"].covars_).tolist()}
    model, states, likelihood = reconstruct_candidate(parameters, observations, 2, "diag")
    assert likelihood == pytest.approx(fitted["log_likelihood"])
    frame = pd.DataFrame({"Log_Return": observations[:, 0], "GARCH_Volatility_TrainFit": observations[:, 1], "Drawdown_252": observations[:, 0] - 3})
    profiles = original_scale_profiles(frame, states, 2)
    assert profiles["count"].sum() == len(frame)
    durations = finalist_duration_summary(states, np.array([[0.9, 0.1], [0.1, 0.9]]), 2)
    assert {"minimum_duration", "maximum_duration", "one_day_episode_percentage"}.issubset(durations.columns)
    assert len(centroid_distances(model.means_)) == 1


def test_alignment_crosstab_ordering_and_external_variable_exclusion():
    profiles = pd.DataFrame({"state": [0, 1], "mean_GARCH_Volatility_TrainFit": [0.01, 0.02],
                             "mean_Drawdown_252": [-0.01, -0.2], "mean_Log_Return": [0.01, -0.01]})
    ordered = ordered_stress_states(profiles)
    assert ordered.loc[ordered["stress_rank"] == 1, "state"].iloc[0] == 0
    cross, mapping = aligned_state_crosstab(np.array([0, 0, 1]), np.array([[0.0], [3.0]]),
                                             np.array([0, 2, 1]), np.array([[0.1], [3.1], [0.2]]))
    assert cross["count"].sum() == 3
    assert set(mapping) == {0, 1, 2}
    with pytest.raises(HMMValidationError):
        validate_features(["Log_Return", "VIX_Close", "Drawdown_252"])


def test_ordered_mapping_and_probability_columns_follow_training_profiles():
    profiles = pd.DataFrame({"state": [0, 1, 2],
        "mean_GARCH_Volatility_TrainFit": [2.0, 1.0, 1.0],
        "mean_Drawdown_252": [-.2, -.3, -.1], "mean_Log_Return": [0., 0., -.1]})
    mapping = ordered_state_mapping(profiles)
    assert mapping == {0: 3, 1: 2, 2: 1}
    reordered = reorder_probabilities([[.2, .3, .5]], mapping)
    assert np.allclose(reordered, [[.5, .3, .2]])


def _assignment_fixture():
    observations = np.vstack([np.zeros((20, 3)), np.full((20, 3), 3.0)])
    fitted = fit_hmm_candidate(observations, 2, "diag", seed=7, n_iter=50)["model"]
    scaler = hmm_model.StandardScaler().fit(observations[:30])
    # Re-express the fitted model for observations transformed by this scaler.
    fitted.means_ = (fitted.means_ - scaler.mean_) / scaler.scale_
    covars = np.diagonal(fitted.covars_, axis1=1, axis2=2) / scaler.scale_**2
    fitted.covars_ = covars
    frame = pd.DataFrame({"Date": pd.date_range("2017-12-01", periods=41),
        "Log_Return": np.r_[observations[:, 0], np.nan],
        "GARCH_Volatility_TrainFit": np.r_[observations[:, 1], 1.],
        "Drawdown_252": np.r_[observations[:, 2], 1.],
        "Abnormal_Volume": 9., "VIX_Close": 8., "GARCH_Volatility_FullSample": 99.})
    return frame, scaler, fitted


def test_selected_assignments_probability_argmax_confidence_and_ineligible_rows():
    frame, scaler, model = _assignment_fixture()
    result = selected_assignments(frame, scaler, model, {0: 1, 1: 2}, "2017-12-31")
    eligible = result.HMM_Eligible
    for kind in ("Filtered", "Smoothed"):
        columns = [f"HMM_{kind}_Probability_State_{x}" for x in (1, 2)]
        values = result.loc[eligible, columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert np.allclose(values.sum(axis=1), 1)
        assert np.array_equal(result.loc[eligible, f"HMM_{kind}_State"].to_numpy(), values.argmax(1) + 1)
        assert np.array_equal(result.loc[eligible, f"HMM_{kind}_Low_Confidence"].to_numpy(), values.max(1) < .60)
    assert result.loc[~eligible, "HMM_Filtered_State"].isna().all()
    assert result.loc[~eligible, "HMM_Smoothed_Probability_State_1"].isna().all()
    assert len(result) == len(frame) and result.Date.is_unique and result.Date.is_monotonic_increasing
    assert set(result.loc[eligible, "HMM_Sample"]) == {"Train", "Test"}


def test_filtering_continues_at_boundary_and_ignores_future_and_external_variables():
    frame, scaler, model = _assignment_fixture()
    base = selected_assignments(frame, scaler, model, {0: 1, 1: 2}, "2017-12-31")
    altered = frame.copy()
    altered.loc[altered.Date > "2017-12-25", FEATURES] = 1_000
    altered["Abnormal_Volume"] = -1e9; altered["VIX_Close"] = 1e9
    changed = selected_assignments(altered, scaler, model, {0: 1, 1: 2}, "2017-12-31")
    columns = [f"HMM_Filtered_Probability_State_{x}" for x in (1, 2)]
    historical = frame.Date <= "2017-12-25"
    assert np.allclose(base.loc[historical, columns], changed.loc[historical, columns])
    external_only = frame.copy(); external_only["Abnormal_Volume"] *= -99; external_only["VIX_Close"] *= 99
    repeated = selected_assignments(external_only, scaler, model, {0: 1, 1: 2}, "2017-12-31")
    pd.testing.assert_frame_equal(base[columns], repeated[columns])
    eligible_x = scaler.transform(frame.loc[frame[FEATURES].notna().all(axis=1), FEATURES])
    uninterrupted = forward_filter(eligible_x, model.startprob_, model.transmat_, model.means_,
        np.diagonal(model.covars_, axis1=1, axis2=2))
    assert np.allclose(base.loc[base.HMM_Eligible, columns], uninterrupted)
    assert not any("Smoothed" in column for column in columns)
