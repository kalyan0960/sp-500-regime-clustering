"""Core, leakage-safe Gaussian-HMM utilities.

This module intentionally contains no model fitting or market-data execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


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


def training_feature_scaler(
    frame: pd.DataFrame, training_end: str | pd.Timestamp
) -> tuple[StandardScaler, pd.DataFrame, np.ndarray]:
    """Fit a scaler only on finite authorized observations through ``training_end``.

    ``Date`` is used when supplied; otherwise the DataFrame index must be datetime-like.
    The returned DataFrame is a copy of only training-eligible observations.
    """
    dates = pd.to_datetime(frame["Date"] if "Date" in frame else frame.index)
    training = frame.loc[dates <= pd.Timestamp(training_end)].copy()
    training = eligible_sample(training)
    if training.empty:
        raise HMMValidationError("No eligible training observations are available.")
    scaler = StandardScaler().fit(training.loc[:, FEATURES])
    return scaler, training, scaler.transform(training.loc[:, FEATURES])


def fit_hmm_candidate(
    observations: np.ndarray,
    k: int,
    covariance_type: str,
    seed: int,
    *,
    n_iter: int = 250,
    tol: float = 1e-3,
    model_factory=GaussianHMM,
) -> dict:
    """Fit one fixed-seed Gaussian-HMM candidate and capture any failure."""
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise HMMValidationError("Candidate observations must be a non-empty finite matrix.")
    if covariance_type not in {"diag", "full"}:
        raise HMMValidationError("covariance_type must be 'diag' or 'full'.")
    record = {"K": k, "covariance_type": covariance_type, "seed": int(seed)}
    try:
        model = model_factory(
            n_components=k,
            covariance_type=covariance_type,
            random_state=seed,
            n_iter=n_iter,
            tol=tol,
            min_covar=1e-6,
            implementation="log",
        )
        model.fit(values)
        history = list(model.monitor_.history)
        final_change = history[-1] - history[-2] if len(history) > 1 else np.nan
        declined = bool(np.isfinite(final_change) and final_change < -tol)
        converged = bool(model.monitor_.converged) and not declined
        log_likelihood = float(model.score(values))
        states = model.predict(values)
        record.update(
            {
                "status": "converged" if converged else "not_converged",
                "converged": converged,
                "log_likelihood": log_likelihood,
                "iterations": int(model.monitor_.iter),
                "convergence_warning": (
                    f"final log-likelihood decreased by {final_change:.6g}" if declined else ""
                ),
                "states": states,
                "model": model,
                "error": "",
            }
        )
    except Exception as error:  # hmmlearn failures are candidate evidence, not silent skips.
        record.update(
            {
                "status": "failed",
                "converged": False,
                "log_likelihood": np.nan,
                "iterations": np.nan,
                "convergence_warning": "",
                "states": None,
                "model": None,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return record


def fit_multiple_seeds(
    observations: np.ndarray,
    k: int,
    covariance_type: str,
    seeds: list[int] | tuple[int, ...],
    **kwargs,
) -> dict:
    """Fit all starts and retain the highest-likelihood converged run only."""
    runs = [fit_hmm_candidate(observations, k, covariance_type, seed, **kwargs) for seed in seeds]
    converged = [run for run in runs if run["converged"]]
    best = max(converged, key=lambda run: run["log_likelihood"], default=None)
    return {"runs": runs, "best_run": best}


def candidate_stability(runs: list[dict]) -> dict[str, float]:
    """Summarize all pairwise ARIs among converged fixed-seed decodings."""
    decodings = [run["states"] for run in runs if run.get("converged")]
    aris = [
        adjusted_rand_score(decodings[left], decodings[right])
        for left in range(len(decodings))
        for right in range(left + 1, len(decodings))
    ]
    if not aris:
        return {"ari_mean": np.nan, "ari_min": np.nan, "ari_std": np.nan, "ari_pairs": 0}
    return {
        "ari_mean": float(np.mean(aris)),
        "ari_min": float(np.min(aris)),
        "ari_std": float(np.std(aris)),
        "ari_pairs": len(aris),
    }


def state_sizes(states: np.ndarray, k: int) -> pd.DataFrame:
    """Return every state count and share, including unoccupied states."""
    values = np.asarray(states, dtype=int)
    counts = np.bincount(values, minlength=k)
    return pd.DataFrame(
        {
            "state": np.arange(k),
            "count": counts,
            "percentage": 100 * counts / len(values),
        }
    )


def transition_matrix(model: GaussianHMM) -> np.ndarray:
    """Return a validated copied transition matrix from a fitted HMM."""
    _, transition = validate_markov(model.startprob_, model.transmat_)
    return transition.copy()


def empirical_duration_summary(states: np.ndarray, k: int) -> pd.DataFrame:
    """Calculate episode counts, duration summaries, and one-day episodes by state."""
    all_episodes = episodes(states)
    rows = []
    for state in range(k):
        durations = [episode["duration"] for episode in all_episodes if episode["state"] == state]
        rows.append(
            {
                "state": state,
                "episode_count": len(durations),
                "empirical_mean_duration": float(np.mean(durations)) if durations else np.nan,
                "empirical_median_duration": float(np.median(durations)) if durations else np.nan,
                "one_day_episode_count": int(np.sum(np.asarray(durations) == 1)),
            }
        )
    return pd.DataFrame(rows)


def degeneracy_warnings(
    states: np.ndarray, k: int, min_state_fraction: float = 0.01
) -> dict[str, object]:
    """Flag unoccupied and small decoded states without changing the candidate."""
    sizes = state_sizes(states, k)
    minimum = int(np.ceil(len(states) * min_state_fraction))
    degenerate = sizes.loc[sizes["count"] == 0, "state"].tolist()
    small = sizes.loc[(sizes["count"] > 0) & (sizes["count"] < minimum), "state"].tolist()
    warnings = []
    if degenerate:
        warnings.append(f"degenerate unoccupied states: {degenerate}")
    if small:
        warnings.append(f"small states below {min_state_fraction:.1%} of training rows: {small}")
    return {
        "small_state_threshold": minimum,
        "degenerate_states": degenerate,
        "small_states": small,
        "warning": "; ".join(warnings),
    }


def covariance_warnings(model: GaussianHMM, tolerance: float = 1e-6) -> dict[str, object]:
    """Flag non-finite, non-positive, or near-singular fitted covariance estimates."""
    covariances = np.asarray(model.covars_, dtype=float)
    warnings: list[str] = []
    minimum_eigenvalues: list[float] = []
    if not np.isfinite(covariances).all():
        warnings.append("non-finite covariance estimate")
    if model.covariance_type == "diag":
        diagonal_values = (
            np.diagonal(covariances, axis1=1, axis2=2)
            if covariances.ndim == 3
            else covariances
        )
        minimum_eigenvalues = np.min(diagonal_values, axis=1).tolist()
    else:
        minimum_eigenvalues = [float(np.min(np.linalg.eigvalsh(matrix))) for matrix in covariances]
    near_singular = [index for index, value in enumerate(minimum_eigenvalues) if value <= tolerance]
    if near_singular:
        warnings.append(f"near-singular covariance states: {near_singular}")
    return {
        "minimum_covariance_eigenvalues": minimum_eigenvalues,
        "near_singular_states": near_singular,
        "warning": "; ".join(warnings),
    }


def summarize_candidate(
    fit_result: dict, observations: np.ndarray, k: int, covariance_type: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Build training-only comparison, state-size, duration, and metadata records."""
    runs, best = fit_result["runs"], fit_result["best_run"]
    stability = candidate_stability(runs)
    base = {
        "K": k,
        "covariance_type": covariance_type,
        "parameter_count": parameter_count(k, observations.shape[1], covariance_type),
        "converged_start_count": sum(run["converged"] for run in runs),
        "nonconverged_start_count": sum(run["status"] == "not_converged" for run in runs),
        "failed_start_count": sum(run["status"] == "failed" for run in runs),
        **stability,
    }
    if best is None:
        base.update({"best_seed": np.nan, "best_training_log_likelihood": np.nan, "BIC": np.nan,
                     "smallest_state_count": np.nan, "smallest_state_percentage": np.nan,
                     "covariance_warning": "", "degeneracy_warning": "no converged run"})
        return base, pd.DataFrame(), pd.DataFrame(), {"best_parameters": None}

    sizes = state_sizes(best["states"], k)
    transition = transition_matrix(best["model"])
    durations = empirical_duration_summary(best["states"], k)
    durations["expected_duration"] = expected_durations(transition)
    covariance = covariance_warnings(best["model"])
    degeneracy = degeneracy_warnings(best["states"], k)
    sizes["K"], sizes["covariance_type"], sizes["best_seed"] = k, covariance_type, best["seed"]
    durations["K"], durations["covariance_type"], durations["best_seed"] = k, covariance_type, best["seed"]
    base.update(
        {
            "best_seed": best["seed"],
            "best_training_log_likelihood": best["log_likelihood"],
            "BIC": bic(best["log_likelihood"], base["parameter_count"], len(observations)),
            "smallest_state_count": int(sizes["count"].min()),
            "smallest_state_percentage": float(sizes["percentage"].min()),
            "transition_matrix": transition.tolist(),
            "diagonal_persistence_probabilities": np.diag(transition).tolist(),
            "covariance_warning": covariance["warning"],
            "degeneracy_warning": degeneracy["warning"],
        }
    )
    metadata = {
        "best_parameters": {
            "seed": best["seed"], "startprob": best["model"].startprob_.tolist(),
            "transmat": transition.tolist(), "means": best["model"].means_.tolist(),
            "covars": np.asarray(best["model"].covars_).tolist(),
        }
    }
    return base, sizes, durations, metadata
