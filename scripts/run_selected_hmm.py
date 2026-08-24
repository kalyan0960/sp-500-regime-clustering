"""Reconstruct the Stage 10C2 selected HMM and write final assignments."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import hmmlearn
import numpy as np
import pandas as pd
import scipy
import sklearn

from market_regime.hmm_model import (
    FEATURES, bic, finalist_duration_summary, ordered_state_mapping,
    original_scale_profiles, reconstruct_candidate, selected_assignments,
    training_feature_scaler, transition_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES, MODELS = ROOT / "outputs" / "tables", ROOT / "outputs" / "models"
K, COVARIANCE, SEED, THRESHOLD = 4, "diag", 44, 0.60


def assignment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = frame.loc[frame["HMM_Eligible"]]
    for sample, group in eligible.groupby("HMM_Sample", sort=False):
        for kind in ("Filtered", "Smoothed"):
            counts = group[f"HMM_{kind}_State"].value_counts().reindex(range(1, K + 1), fill_value=0)
            for state, count in counts.items():
                rows.append({"sample": sample, "metric": f"{kind.lower()}_state", "state": state,
                             "count": int(count), "percentage": 100 * count / len(group)})
            low = int(group[f"HMM_{kind}_Low_Confidence"].sum())
            rows.append({"sample": sample, "metric": f"{kind.lower()}_low_confidence", "state": pd.NA,
                         "count": low, "percentage": 100 * low / len(group)})
        agreement = int((group["HMM_Filtered_State"] == group["HMM_Smoothed_State"]).sum())
        rows.append({"sample": sample, "metric": "filtered_smoothed_agreement", "state": pd.NA,
                     "count": agreement, "percentage": 100 * agreement / len(group)})
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    source_metadata = json.loads((MODELS / "hmm_candidate_metadata.json").read_text())
    frame = pd.read_csv(ROOT / source_metadata["input"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    scaler, training, training_x = training_feature_scaler(frame, source_metadata["training_end"])
    if not (np.allclose(scaler.mean_, source_metadata["scaler_mean"]) and
            np.allclose(scaler.scale_, source_metadata["scaler_scale"])):
        raise RuntimeError("Training-only scaler differs from Stage 10B metadata.")
    parameters = source_metadata["candidates"]["K4_diag"]["best_parameters"]
    if int(parameters["seed"]) != SEED:
        raise RuntimeError("Saved selected-model seed is not 44.")
    model, training_states, likelihood = reconstruct_candidate(parameters, training_x, K, COVARIANCE)
    comparison = pd.read_csv(TABLES / "hmm_model_comparison.csv")
    source = comparison.query("K == @K and covariance_type == @COVARIANCE").iloc[0]
    calculated_bic = bic(likelihood, int(source["parameter_count"]), len(training_x))
    if not np.isclose(likelihood, source["best_training_log_likelihood"], atol=1e-6):
        raise RuntimeError("Reconstructed likelihood differs from Stage 10B.")
    if not np.isclose(calculated_bic, source["BIC"], atol=1e-6):
        raise RuntimeError("Reconstructed BIC differs from Stage 10B.")

    profiles = original_scale_profiles(training, training_states, K)
    mapping = ordered_state_mapping(profiles)
    profiles["Ordered_State"] = profiles["state"].map(mapping)
    profiles["Original_State"] = profiles.pop("state")
    profiles["Neutral_State_Label"] = "Ordered_State_" + profiles["Ordered_State"].astype(str)
    profiles = profiles.sort_values("Ordered_State")
    profiles.to_csv(TABLES / "hmm_selected_model_profile.csv", index=False)

    merged = selected_assignments(frame, scaler, model, mapping, source_metadata["training_end"], THRESHOLD)
    merged.to_csv(TABLES / "market_with_hmm.csv", index=False)
    assignment_summary(merged).to_csv(TABLES / "hmm_assignment_summary.csv", index=False)

    transition = transition_matrix(model)
    original_by_order = [next(old for old, new in mapping.items() if new == state) for state in range(1, K + 1)]
    ordered_transition = transition[np.ix_(original_by_order, original_by_order)]
    transition_table = pd.DataFrame(ordered_transition, columns=[f"Ordered_State_{x}" for x in range(1, K + 1)])
    transition_table.insert(0, "From_State", [f"Ordered_State_{x}" for x in range(1, K + 1)])
    transition_table["Row_Sum"] = ordered_transition.sum(axis=1)
    transition_table.to_csv(TABLES / "hmm_selected_transition_matrix.csv", index=False)

    durations = finalist_duration_summary(training_states, transition, K)
    durations["Ordered_State"] = durations["state"].map(mapping)
    durations["Original_State"] = durations.pop("state")
    durations = durations.rename(columns={"expected_duration": "model_implied_expected_duration"}).sort_values("Ordered_State")
    durations.to_csv(TABLES / "hmm_selected_state_durations.csv", index=False)

    metadata = {
        "K": K, "covariance_type": COVARIANCE, "seed": SEED, "features_in_order": FEATURES,
        "training_cutoff": source_metadata["training_end"], "reconstruction_method": "saved Stage 10B parameters; no refit",
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "original_to_ordered_state": {str(k): v for k, v in mapping.items()},
        "stage_10b_log_likelihood": float(source["best_training_log_likelihood"]),
        "reconstructed_log_likelihood": likelihood, "stage_10b_BIC": float(source["BIC"]),
        "reconstructed_BIC": calculated_bic, "confidence_threshold": THRESHOLD,
        "filtering_method": "log-space forward recursion over the entire eligible chronology without train/test reset",
        "smoothed_method": "retrospective full-eligible-sequence posterior; prohibited as forecasting predictor",
        "leakage_restrictions": "Training scaler and fixed HMM parameters only; excludes volume, abnormal volume, VIX, future returns, event labels, test fitting, and full-sample GARCH.",
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "hmmlearn": hmmlearn.__version__},
    }
    (MODELS / "hmm_selected_model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"rows": len(merged), "eligible": int(merged.HMM_Eligible.sum()), "mapping": mapping,
                      "log_likelihood": likelihood, "BIC": calculated_bic}, indent=2))


if __name__ == "__main__":
    main()
