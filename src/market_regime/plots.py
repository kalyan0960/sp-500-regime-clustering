"""Publication plotting helpers for the finalized Stage 10C2 HMM outputs."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STATE_COLORS = {1: "#2A9D8F", 2: "#457B9D", 3: "#F4A261", 4: "#D1495B"}
STATE_LABELS = {s: f"Ordered State {s}" for s in STATE_COLORS}
CUTOFF = pd.Timestamp("2017-12-31")


def set_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 180, "font.family": "DejaVu Sans",
                         "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})


def _save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _cutoff(ax: plt.Axes) -> None:
    ax.axvline(CUTOFF, color="#333333", linestyle="--", linewidth=1.2, label="Training cutoff")


def _dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["Date"])


def plot_filtered_price(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5)); dates = _dates(frame)
    ax.plot(dates, frame["Adj_Close"], color="#B8B8B8", linewidth=.65, zorder=1)
    for state, color in STATE_COLORS.items():
        mask = frame["HMM_Filtered_State"].eq(state)
        ax.scatter(dates[mask], frame.loc[mask, "Adj_Close"], s=5, color=color,
                   label=STATE_LABELS[state], rasterized=True, zorder=2)
    _cutoff(ax); ax.axvspan(CUTOFF, dates.max(), color="#457B9D", alpha=.035, zorder=0)
    ax.set(title="SPY Adjusted Price by Forward-Filtered HMM Regime", ylabel="Adjusted price (USD)", xlabel="Date")
    ax.legend(ncol=3, loc="upper left")
    for text, date, price, xytext in [
        ("Global Financial Crisis", "2008-09-15", 90, ("2005-06-01", 155)),
        ("COVID-19 shock", "2020-03-16", 223, ("2018-06-01", 390)),
        ("2022 decline", "2022-06-15", 365, ("2023-01-01", 275))]:
        ax.annotate(text, (pd.Timestamp(date), price), xytext=(pd.Timestamp(xytext[0]), xytext[1]),
                    arrowprops={"arrowstyle": "-", "color": ".4"})
    _save(fig, output)


def plot_filtered_features(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True); dates = _dates(frame)
    specs = [("Log_Return", "Log return", "Daily log return"),
             ("GARCH_Volatility_TrainFit", "Conditional volatility", "Train-fitted GARCH volatility"),
             ("Drawdown_252", "252-day drawdown", "Drawdown from rolling peak")]
    for ax, (column, title, ylabel) in zip(axes, specs):
        ax.plot(dates, frame[column], color="#CCCCCC", linewidth=.45)
        for state, color in STATE_COLORS.items():
            mask = frame["HMM_Filtered_State"].eq(state)
            ax.scatter(dates[mask], frame.loc[mask, column], s=2.5, color=color, rasterized=True)
        _cutoff(ax); ax.set_title(title, loc="left", fontsize=10); ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Date")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=STATE_LABELS[s]) for s, c in STATE_COLORS.items()]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Market Features Colored by Forward-Filtered HMM Regime", y=1.04, fontweight="bold")
    _save(fig, output)


def plot_probabilities(frame: pd.DataFrame, kind: str, output: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, sharey=True); dates = _dates(frame)
    for state, ax in zip(STATE_COLORS, axes):
        values = frame[f"HMM_{kind}_Probability_State_{state}"]
        ax.plot(dates, values, color=STATE_COLORS[state], linewidth=.75)
        ax.fill_between(dates, 0, values, color=STATE_COLORS[state], alpha=.12)
        _cutoff(ax); ax.set_ylim(-.02, 1.02); ax.set_ylabel(f"State {state}")
    qualifier = "Real-Time / Forward-Filtered" if kind == "Filtered" else "Retrospective / Full-Sequence Smoothed"
    fig.suptitle(f"{qualifier} HMM State Probabilities", fontweight="bold"); axes[-1].set_xlabel("Date")
    _save(fig, output)


def plot_transition_matrix(transition: pd.DataFrame, output: Path) -> None:
    matrix = transition.filter(regex=r"^Ordered_State_\d$").to_numpy(float)
    labels = [f"State {s}" for s in STATE_COLORS]
    annotations = np.vectorize(lambda x: f"{x:.4f}" if x >= 1e-4 else (f"{x:.1e}" if x > 0 else "0 (saved)"))(matrix)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(matrix, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, square=True,
                cbar_kws={"label": "Transition probability"}, ax=ax)
    ax.set(title="Selected HMM Transition Matrix", xlabel="To state", ylabel="From state"); _save(fig, output)


def plot_profiles(profiles: pd.DataFrame, metadata: dict, output: Path) -> None:
    standardized = pd.DataFrame({"Ordered State": profiles["Ordered_State"].astype(int),
        "Log return": (profiles["mean_Log_Return"] - metadata["scaler_mean"][0]) / metadata["scaler_scale"][0],
        "GARCH volatility": (profiles["mean_GARCH_Volatility_TrainFit"] - metadata["scaler_mean"][1]) / metadata["scaler_scale"][1],
        "252-day drawdown": (profiles["mean_Drawdown_252"] - metadata["scaler_mean"][2]) / metadata["scaler_scale"][2]}).melt(
            "Ordered State", var_name="Feature", value_name="Training-standardized mean")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=standardized, x="Feature", y="Training-standardized mean", hue="Ordered State",
                palette=STATE_COLORS, ax=ax); ax.axhline(0, color=".25", linewidth=.8)
    ax.set(title="Training-Period HMM State Profiles", xlabel="Feature",
           ylabel="Mean in training-scaler standard deviations"); ax.legend(title="Ordered state", ncol=4)
    _save(fig, output)


def plot_durations(durations: pd.DataFrame, output: Path) -> None:
    long = durations.rename(columns={"model_implied_expected_duration": "Model-implied expected",
        "empirical_mean_duration": "Empirical mean", "empirical_median_duration": "Empirical median"}).melt(
        id_vars="Ordered_State", value_vars=["Model-implied expected", "Empirical mean", "Empirical median"],
        var_name="Duration measure", value_name="Trading observations")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=long, x="Ordered_State", y="Trading observations", hue="Duration measure",
                palette=["#457B9D", "#E9C46A", "#8D99AE"], ax=ax)
    ax.set(title="Model-Implied and Empirical Training Episode Durations", xlabel="Ordered state",
           ylabel="Duration (trading observations)"); _save(fig, output)


def plot_train_test(summary: pd.DataFrame, output: Path) -> None:
    data = summary.loc[summary["metric"].eq("filtered_state")].copy(); data["state"] = data["state"].astype(int)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.barplot(data=data, x="state", y="percentage", hue="sample", palette=["#6C757D", "#457B9D"], ax=ax)
    ax.set(title="Forward-Filtered Regime Composition: Training vs Testing", xlabel="Ordered state",
           ylabel="Share of eligible observations (%)"); ax.legend(title="Sample"); _save(fig, output)


def plot_confidence(frame: pd.DataFrame, threshold: float, output: Path) -> None:
    data = frame.loc[frame["HMM_Eligible"], ["HMM_Sample", "HMM_Filtered_Max_Probability"]]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for sample, label, color in (("Train", "Training", "#457B9D"),
                                 ("Test", "Testing", "#F4A261")):
        sns.histplot(data=data.loc[data["HMM_Sample"].eq(sample)],
                     x="HMM_Filtered_Max_Probability", bins=35, stat="density",
                     element="step", fill=False, linewidth=1.6, color=color,
                     label=label, ax=ax)
    ax.axvline(threshold, color="#D1495B", linestyle="--", linewidth=1.5,
               label=f"Low-confidence threshold ({threshold:.2f})")
    ax.set(title="Forward-Filtered Classification Confidence", xlabel="Maximum filtered state probability",
           ylabel="Density", xlim=(.25, 1.01)); ax.legend(); _save(fig, output)


def plot_agreement(frame: pd.DataFrame, output: Path) -> None:
    eligible = frame.loc[frame["HMM_Eligible"]]
    cross = pd.crosstab(eligible["HMM_Filtered_State"], eligible["HMM_Smoothed_State"], normalize="index")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(cross, annot=True, fmt=".1%", cmap="YlGnBu", vmin=0, vmax=1, square=True,
                cbar_kws={"label": "Row-normalized share"}, ax=ax)
    ax.set(title="Filtered–Smoothed State Agreement (Diagnostic)", xlabel="Retrospective smoothed state",
           ylabel="Forward-filtered state"); _save(fig, output)


def create_hmm_diagnostic_figures(frame: pd.DataFrame, profiles: pd.DataFrame,
                                  transition: pd.DataFrame, durations: pd.DataFrame,
                                  summary: pd.DataFrame, metadata: dict, output_dir: Path) -> list[Path]:
    """Create ten Stage 10D figures solely from saved Stage 10C2 outputs."""
    set_publication_style(); output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("hmm_filtered_regimes_price.png", plot_filtered_price, (frame,)),
            ("hmm_filtered_regime_features.png", plot_filtered_features, (frame,)),
            ("hmm_filtered_probabilities.png", plot_probabilities, (frame, "Filtered")),
            ("hmm_smoothed_probabilities.png", plot_probabilities, (frame, "Smoothed")),
            ("hmm_transition_matrix.png", plot_transition_matrix, (transition,)),
            ("hmm_state_profiles.png", plot_profiles, (profiles, metadata)),
            ("hmm_state_durations.png", plot_durations, (durations,)),
            ("hmm_train_test_distributions.png", plot_train_test, (summary,)),
            ("hmm_classification_confidence.png", plot_confidence, (frame, metadata["confidence_threshold"])),
            ("hmm_filtered_smoothed_agreement.png", plot_agreement, (frame,))]
    paths = []
    for name, function, args in jobs:
        path = output_dir / name; function(*args, path); paths.append(path)
    return paths
