"""Deterministic unit tests for Stage 10A HMM utility safeguards."""

import numpy as np
import pandas as pd
import pytest

from market_regime.hmm_model import (
    FEATURES,
    HMMValidationError,
    bic,
    eligible_sample,
    episodes,
    expected_durations,
    filtered_summary,
    forward_filter,
    parameter_count,
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
