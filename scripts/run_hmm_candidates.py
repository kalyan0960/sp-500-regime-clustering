"""Run the Stage 10B training-only Gaussian-HMM candidate comparison.

Usage: .\\.venv312\\Scripts\\python.exe scripts\\run_hmm_candidates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime.hmm_model import (
    FEATURES,
    fit_multiple_seeds,
    summarize_candidate,
    training_feature_scaler,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "market_with_kmeans.csv"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "outputs" / "models"
SEEDS = list(range(42, 62))


def main() -> None:
    frame = pd.read_csv(INPUT)
    scaler, training, observations = training_feature_scaler(frame, "2017-12-31")
    comparison, sizes, durations, convergence = [], [], [], []
    metadata = {
        "input": str(INPUT.relative_to(ROOT)),
        "features": FEATURES,
        "training_end": "2017-12-31",
        "eligible_training_observations": len(training),
        "seeds": SEEDS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "candidates": {},
    }
    for k in (2, 3, 4, 5):
        for covariance_type in ("diag", "full"):
            result = fit_multiple_seeds(observations, k, covariance_type, SEEDS)
            summary, state_sizes, duration_summary, candidate_metadata = summarize_candidate(
                result, observations, k, covariance_type
            )
            comparison.append(summary)
            if not state_sizes.empty:
                sizes.append(state_sizes)
            if not duration_summary.empty:
                durations.append(duration_summary)
            convergence.extend(
                {
                    "K": run["K"], "covariance_type": run["covariance_type"],
                    "seed": run["seed"], "status": run["status"],
                    "converged": run["converged"], "log_likelihood": run["log_likelihood"],
                    "iterations": run["iterations"], "convergence_warning": run["convergence_warning"],
                    "error": run["error"],
                }
                for run in result["runs"]
            )
            metadata["candidates"][f"K{k}_{covariance_type}"] = candidate_metadata

    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(TABLES / "hmm_model_comparison.csv", index=False)
    pd.concat(sizes, ignore_index=True).to_csv(TABLES / "hmm_candidate_state_sizes.csv", index=False)
    pd.concat(durations, ignore_index=True).to_csv(TABLES / "hmm_candidate_durations.csv", index=False)
    comparison_frame.loc[:, ["K", "covariance_type", "ari_mean", "ari_min", "ari_std", "ari_pairs"]].to_csv(
        TABLES / "hmm_candidate_stability.csv", index=False
    )
    pd.DataFrame(convergence).to_csv(TABLES / "hmm_candidate_convergence.csv", index=False)
    (MODELS / "hmm_candidate_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
