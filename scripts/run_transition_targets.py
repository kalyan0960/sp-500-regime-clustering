"""Build Stage 11 filtered-state future targets and descriptive diagnostics."""
from pathlib import Path
import pandas as pd

from market_regime.plots import create_transition_diagnostic_figures
from market_regime.transitions import (construct_transition_targets, filtered_episode_summary,
    filtered_episodes, filtered_transition_counts, transition_confidence_summary, transition_target_summary)

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = ROOT/"outputs"/"tables", ROOT/"outputs"/"figures"


def main() -> None:
    source = pd.read_csv(TABLES/"market_with_hmm.csv")
    targets = construct_transition_targets(source)
    summary = transition_target_summary(targets)
    counts = filtered_transition_counts(targets)
    episodes = filtered_episodes(targets)
    episode_summary = filtered_episode_summary(episodes)
    confidence = transition_confidence_summary(targets)
    targets.to_csv(TABLES/"market_with_transition_targets.csv", index=False)
    summary.to_csv(TABLES/"transition_target_summary.csv", index=False)
    counts.to_csv(TABLES/"filtered_transition_counts.csv", index=False)
    episode_summary.to_csv(TABLES/"filtered_episode_summary.csv", index=False)
    confidence.to_csv(TABLES/"transition_confidence_summary.csv", index=False)
    create_transition_diagnostic_figures(targets, summary, counts, episodes, confidence, FIGURES)
    print(f"rows={len(targets)}, added_columns={len(targets.columns)-len(source.columns)}")


if __name__ == "__main__": main()
