"""Core, leakage-safe Gaussian-HMM utilities.

This module intentionally contains no model fitting or market-data execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


HMM_FEATURES = (
    "Log_Return",
    "GARCH_Volatility_TrainFit",
    "Drawdown_252",
)
# Backward-compatible public name for the approved ordered feature list.
FEATURES = list(HMM_FEATURES)


class HMMValidationError(ValueError):
    """Raised when fixed HMM inputs do not meet the research-design contract."""


def validate_features(features: list[str] | tuple[str, ...]) -> list[str]:
    """Require the exact authorized ordered HMM feature list."""
    if list(features) != FEATURES:
        raise HMMValidationError(
            "HMM features must be exactly Log_Return, "
            "GARCH_Volatility_TrainFit, Drawdown_252."
        )
    return FEATURES.copy()


def eligible_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copied, finite HMM-eligible subset without changing ``frame``."""
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise HMMValidationError(f"Missing required HMM features: {missing}.")
    eligible = np.isfinite(frame.loc[:, FEATURES]).all(axis=1)
    return frame.loc[eligible].copy()


def parameter_count(k: int, d: int, covariance_type: str) -> int:
    """Return Gaussian-HMM free parameters for ``k`` states and ``d`` features."""
    if not isinstance(k, int) or not isinstance(d, int) or k < 1 or d < 1:
        raise HMMValidationError("k and d must be positive integers.")
    if covariance_type == "diag":
        covariance_parameters = k * d
    elif covariance_type == "full":
        covariance_parameters = k * d * (d + 1) // 2
    else:
        raise HMMValidationError("covariance_type must be 'diag' or 'full'.")
    return (k - 1) + k * (k - 1) + k * d + covariance_parameters


def bic(log_likelihood: float, parameter_count_value: int, n: int) -> float:
    """Calculate BIC = -2 log L + q log n for likelihood ``L`` and q parameters."""
    if n < 1 or parameter_count_value < 0:
        raise HMMValidationError("n must be positive and parameter count non-negative.")
    return float(-2 * log_likelihood + parameter_count_value * np.log(n))


def validate_markov(
    start_probabilities: np.ndarray, transition_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return a fixed finite start vector and row-stochastic matrix."""
    start = np.asarray(start_probabilities, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)
    k = start.size
    valid = (
        start.ndim == 1
        and k > 0
        and transition.shape == (k, k)
        and np.isfinite(start).all()
        and np.isfinite(transition).all()
        and (start >= 0).all()
        and (transition >= 0).all()
        and np.isclose(start.sum(), 1.0)
        and np.allclose(transition.sum(axis=1), 1.0)
    )
    if not valid:
        raise HMMValidationError(
            "Start probabilities must sum to one and transition rows must sum to one."
        )
    return start, transition


def expected_durations(transition_matrix: np.ndarray) -> np.ndarray:
    """Return state durations 1 / (1 - p_ii) implied by a transition matrix."""
    transition = np.asarray(transition_matrix, dtype=float)
    _, transition = validate_markov(
        np.full(transition.shape[0], 1.0 / transition.shape[0]), transition
    )
    diagonal = np.diag(transition)
    if np.any(diagonal >= 1.0):
        raise HMMValidationError("Expected duration is infinite when p_ii equals one.")
    return 1.0 / (1.0 - diagonal)


def episodes(states: list[int] | np.ndarray) -> list[dict[str, int]]:
    """Extract contiguous state episodes and their empirical observation durations."""
    sequence = list(states)
    if not sequence:
        return []
    result: list[dict[str, int]] = []
    start = 0
    for position in range(1, len(sequence) + 1):
        if position == len(sequence) or sequence[position] != sequence[start]:
            result.append(
                {
                    "state": int(sequence[start]),
                    "start_index": start,
                    "end_index": position - 1,
                    "duration": position - start,
                }
            )
            start = position
    return result


def _log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to logs without emitting warnings for valid zeros."""
    logged = np.full(probabilities.shape, -np.inf, dtype=float)
    positive = probabilities > 0
    logged[positive] = np.log(probabilities[positive])
    return logged


def forward_filter(
    observations: np.ndarray,
    start_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    covariance_type: str = "diag",
) -> np.ndarray:
    """Filter fixed Gaussian-HMM state probabilities using observations through t only."""
    start, transition = validate_markov(start_probabilities, transition_matrix)
    values = np.asarray(observations, dtype=float)
    state_means = np.asarray(means, dtype=float)
    covariance_values = np.asarray(covariances, dtype=float)
    k = len(start)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise HMMValidationError("observations must be a non-empty finite two-dimensional array.")
    if state_means.shape != (k, values.shape[1]):
        raise HMMValidationError("means must have shape (number of states, features).")
    if covariance_type == "diag":
        if covariance_values.shape != state_means.shape or np.any(covariance_values <= 0):
            raise HMMValidationError("Diagonal covariances must be positive with shape (states, features).")
        state_covariances = [np.diag(row) for row in covariance_values]
    elif covariance_type == "full":
        expected_shape = (k, values.shape[1], values.shape[1])
        if covariance_values.shape != expected_shape:
            raise HMMValidationError("Full covariances must have shape (states, features, features).")
        state_covariances = list(covariance_values)
    else:
        raise HMMValidationError("covariance_type must be 'diag' or 'full'.")

    filtered_log_probabilities = np.empty((len(values), k))
    log_prior = _log_probabilities(start)
    log_transition = _log_probabilities(transition)
    for time, observation in enumerate(values):
        log_emission = np.array(
            [
                multivariate_normal.logpdf(
                    observation, mean=state_means[state], cov=state_covariances[state]
                )
                for state in range(k)
            ]
        )
        unnormalized = log_prior + log_emission
        filtered_log_probabilities[time] = unnormalized - logsumexp(unnormalized)
        log_prior = logsumexp(
            filtered_log_probabilities[time][:, None] + log_transition, axis=0
        )
    return np.exp(filtered_log_probabilities)


def filtered_summary(probabilities: np.ndarray, threshold: float = 0.60) -> pd.DataFrame:
    """Return filtered state, confidence, and the strict low-confidence indicator."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.allclose(values.sum(axis=1), 1.0):
        raise HMMValidationError("probabilities must be a row-normalized two-dimensional matrix.")
    maximum = values.max(axis=1)
    return pd.DataFrame(
        {
            "HMM_Filtered_State_Original": values.argmax(axis=1),
            "HMM_Filtered_Max_Probability": maximum,
            "HMM_Low_Confidence": maximum < threshold,
        }
    )
