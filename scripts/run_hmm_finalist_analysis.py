"""Run Stage 10C1 finalist reconstruction and training-only evidence tables.

Usage: .\\.venv312\\Scripts\\python.exe scripts\\run_hmm_finalist_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime.hmm_model import (
    aligned_state_crosstab, bic, centroid_distances, covariance_warnings,
    finalist_duration_summary, original_scale_profiles, ordered_stress_states,
    reconstruct_candidate, training_feature_scaler, transition_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES, MODELS = ROOT / "outputs" / "tables", ROOT / "outputs" / "models"
FINALISTS = [(3, "full"), (4, "diag"), (4, "full"), (5, "full")]


def key(k: int, covariance_type: str) -> str:
    return f"K{k}_{covariance_type}"


def main() -> None:
    metadata = json.loads((MODELS / "hmm_candidate_metadata.json").read_text())
    frame = pd.read_csv(ROOT / metadata["input"])
    scaler, training, observations = training_feature_scaler(frame, metadata["training_end"])
    if not (np.allclose(scaler.mean_, metadata["scaler_mean"]) and np.allclose(scaler.scale_, metadata["scaler_scale"])):
        raise RuntimeError("Training-only scaler differs from saved Stage 10B metadata.")
    comparison = pd.read_csv(TABLES / "hmm_model_comparison.csv")
    profiles_all, durations_all, distances_all, transitions_all, summaries = [], [], [], [], []
    reconstructed: dict[str, dict] = {}
    for k, covariance_type in FINALISTS:
        candidate_key = key(k, covariance_type)
        saved = metadata["candidates"][candidate_key]["best_parameters"]
        model, states, likelihood = reconstruct_candidate(saved, observations, k, covariance_type)
        source = comparison.query("K == @k and covariance_type == @covariance_type").iloc[0]
        if not np.isclose(likelihood, source["best_training_log_likelihood"], atol=1e-6):
            raise RuntimeError(f"{candidate_key} reconstruction likelihood differs from Stage 10B.")
        calculated_bic = bic(likelihood, int(source["parameter_count"]), len(observations))
        if not np.isclose(calculated_bic, source["BIC"], atol=1e-6):
            raise RuntimeError(f"{candidate_key} reconstruction BIC differs from Stage 10B.")
        transition = transition_matrix(model)
        profiles = ordered_stress_states(original_scale_profiles(training, states, k))
        profiles.insert(0, "covariance_type", covariance_type)
        profiles.insert(0, "K", k)
        durations = finalist_duration_summary(states, transition, k).merge(
            profiles[["state", "stress_rank", "ordered_state_name"]], on="state", how="left"
        )
        durations.insert(0, "covariance_type", covariance_type)
        durations.insert(0, "K", k)
        distances = centroid_distances(model.means_)
        distances.insert(0, "covariance_type", covariance_type)
        distances.insert(0, "K", k)
        transition_long = pd.DataFrame(transition).rename_axis("from_state").rename_axis("to_state", axis=1).stack().rename("probability").reset_index()
        transition_long.insert(0, "covariance_type", covariance_type)
        transition_long.insert(0, "K", k)
        covariance = covariance_warnings(model)
        summaries.append({
            "K": k, "covariance_type": covariance_type, "best_seed": int(source["best_seed"]),
            "best_training_log_likelihood": likelihood, "BIC": calculated_bic,
            "converged_start_count": int(source["converged_start_count"]),
            "nonconverged_start_count": int(source["nonconverged_start_count"]),
            "failed_start_count": int(source["failed_start_count"]),
            "ari_mean": source["ari_mean"], "ari_min": source["ari_min"], "ari_std": source["ari_std"],
            "smallest_state_count": int(profiles["count"].min()),
            "smallest_state_percentage": profiles["percentage"].min(),
            "covariance_warning": covariance["warning"],
            "minimum_covariance_eigenvalues": json.dumps(covariance["minimum_covariance_eigenvalues"]),
            "reconstruction_verified": True,
        })
        profiles_all.append(profiles); durations_all.append(durations); distances_all.append(distances); transitions_all.append(transition_long)
        reconstructed[candidate_key] = {"states": states, "means": model.means_, "profiles": profiles}

    profiles_frame = pd.concat(profiles_all, ignore_index=True)
    profiles_frame.to_csv(TABLES / "hmm_finalist_profiles.csv", index=False)
    pd.concat(durations_all, ignore_index=True).to_csv(TABLES / "hmm_finalist_durations.csv", index=False)
    pd.concat(distances_all, ignore_index=True).to_csv(TABLES / "hmm_finalist_centroid_distances.csv", index=False)
    pd.concat(transitions_all, ignore_index=True).to_csv(TABLES / "hmm_finalist_transition_matrices.csv", index=False)
    pd.DataFrame(summaries).to_csv(TABLES / "hmm_finalist_selection_summary.csv", index=False)

    for output, lower_key, higher_key in [
        ("hmm_k3_to_k4_crosstab.csv", "K3_full", "K4_full"),
        ("hmm_k4_to_k5_crosstab.csv", "K4_full", "K5_full"),
    ]:
        cross, mapping = aligned_state_crosstab(
            reconstructed[lower_key]["states"], reconstructed[lower_key]["means"],
            reconstructed[higher_key]["states"], reconstructed[higher_key]["means"],
        )
        cross["alignment_method"] = "Hungarian closest-centroid match; extra higher-K state assigned to nearest lower-K centroid"
        cross.to_csv(TABLES / output, index=False)
        print(f"{lower_key} to {higher_key} alignment: {mapping}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
