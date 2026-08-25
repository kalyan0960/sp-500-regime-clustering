"""Leakage-safe future targets and diagnostics from filtered HMM states."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)
TARGET_BASES = ("Any_Transition_Within", "Endpoint_Transition", "Stress_Entry_Within",
                "Stress_Exit_Within", "Acute_Entry_Within", "Stress_Worsening_Within",
                "Stress_Improving_Within")
LOWER, HIGHER = {1, 2}, {3, 4}


def target_columns(h: int) -> list[str]:
    return [f"{base}_{h}D" for base in TARGET_BASES]


def _nullable_int(n: int) -> pd.Series:
    return pd.Series(pd.NA, index=range(n), dtype="Int64")


def construct_transition_targets(frame: pd.DataFrame, cutoff="2017-12-31",
                                 horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Add future-window outcomes using only ordered forward-filtered states."""
    required = {"Date", "HMM_Eligible", "HMM_Filtered_State", "HMM_Filtered_Low_Confidence"}
    missing = required.difference(frame.columns)
    if missing: raise ValueError(f"Missing transition inputs: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True); dates = pd.to_datetime(result["Date"])
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and chronological.")
    states = pd.array(result["HMM_Filtered_State"], dtype="Int64")
    eligible = result["HMM_Eligible"].fillna(False).astype(bool).to_numpy()
    confidence = ~result["HMM_Filtered_Low_Confidence"].fillna(True).astype(bool).to_numpy()
    valid = eligible & ~pd.isna(states); state_values = np.asarray(states.fillna(-1), dtype=int)
    n, cutoff = len(result), pd.Timestamp(cutoff)
    result["Current_State_Low_Confidence"] = result["HMM_Filtered_Low_Confidence"].astype("boolean")
    for h in horizons:
        targets = {column: _nullable_int(n) for column in target_columns(h)}
        details = {name: _nullable_int(n) for name in (f"First_Transition_Lead_{h}D",
            f"First_Transition_Destination_{h}D", f"State_At_Horizon_{h}D",
            f"Maximum_State_Within_{h}D", f"Minimum_State_Within_{h}D",
            f"Number_of_State_Changes_Within_{h}D")}
        samples = pd.Series("Unavailable", index=range(n), dtype="string")
        future_low = pd.Series(pd.NA, index=range(n), dtype="boolean")
        all_high = pd.Series(pd.NA, index=range(n), dtype="boolean")
        for i in range(n):
            if not valid[i] or i + h >= n or not valid[i + 1:i + h + 1].all(): continue
            window, current = state_values[i + 1:i + h + 1], state_values[i]
            samples.iloc[i] = ("Train" if dates.iloc[i + h] <= cutoff else "Purged") if dates.iloc[i] <= cutoff else "Test"
            low = bool((~confidence[i + 1:i + h + 1]).any()); future_low.iloc[i] = low
            all_high.iloc[i] = bool(confidence[i] and not low)
            changes = np.flatnonzero(window != current)
            targets[f"Any_Transition_Within_{h}D"].iloc[i] = int(len(changes) > 0)
            targets[f"Endpoint_Transition_{h}D"].iloc[i] = int(window[-1] != current)
            if current in LOWER: targets[f"Stress_Entry_Within_{h}D"].iloc[i] = int(np.isin(window, list(HIGHER)).any())
            if current in HIGHER: targets[f"Stress_Exit_Within_{h}D"].iloc[i] = int(np.isin(window, list(LOWER)).any())
            if current != 4:
                targets[f"Acute_Entry_Within_{h}D"].iloc[i] = int((window == 4).any())
                targets[f"Stress_Worsening_Within_{h}D"].iloc[i] = int((window > current).any())
            if current != 1: targets[f"Stress_Improving_Within_{h}D"].iloc[i] = int((window < current).any())
            if len(changes):
                lead = int(changes[0] + 1); details[f"First_Transition_Lead_{h}D"].iloc[i] = lead
                details[f"First_Transition_Destination_{h}D"].iloc[i] = int(window[lead - 1])
            details[f"State_At_Horizon_{h}D"].iloc[i] = int(window[-1])
            details[f"Maximum_State_Within_{h}D"].iloc[i] = int(window.max())
            details[f"Minimum_State_Within_{h}D"].iloc[i] = int(window.min())
            full = np.r_[current, window]
            details[f"Number_of_State_Changes_Within_{h}D"].iloc[i] = int(np.sum(full[1:] != full[:-1]))
        result[f"Transition_Sample_{h}D"] = samples
        result[f"Future_Window_Contains_Low_Confidence_{h}D"] = future_low
        result[f"Transition_Window_All_High_Confidence_{h}D"] = all_high
        for name, values in {**targets, **details}.items(): result[name] = values
    return result


def transition_target_summary(frame: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sample_col = f"Transition_Sample_{h}D"
        for target in target_columns(h):
            for sample in ("All", "Train", "Test"):
                scope = frame if sample == "All" else frame.loc[frame[sample_col].eq(sample)]
                available = int(scope[target].notna().sum()); positives = int(scope[target].eq(1).sum())
                complete = scope[sample_col].isin(["Train", "Test", "Purged"])
                rows.append({"horizon": h, "target": target, "sample": sample,
                    "available_observations": available, "positives": positives,
                    "negatives": int(scope[target].eq(0).sum()), "event_rate": positives / available if available else np.nan,
                    "structurally_inapplicable_count": int((complete & scope[target].isna()).sum()),
                    "unavailable_count": int(scope[sample_col].eq("Unavailable").sum()),
                    "purged_count": int(scope[sample_col].eq("Purged").sum())})
    return pd.DataFrame(rows)


def filtered_transition_counts(frame: pd.DataFrame, cutoff="2017-12-31") -> pd.DataFrame:
    dates, cutoff = pd.to_datetime(frame["Date"]), pd.Timestamp(cutoff)
    valid = frame["HMM_Eligible"].fillna(False) & frame["HMM_Filtered_State"].notna(); rows = []
    for i in range(len(frame) - 1):
        if not (valid.iloc[i] and valid.iloc[i + 1]): continue
        sample = "Train" if dates.iloc[i + 1] <= cutoff else ("Test" if dates.iloc[i] > cutoff else "Crosses_Cutoff")
        rows.append((sample, int(frame.HMM_Filtered_State.iloc[i]), int(frame.HMM_Filtered_State.iloc[i + 1])))
    raw = pd.DataFrame(rows, columns=["Sample", "From_State", "To_State"]); outputs = []
    for sample in ("All", "Train", "Test", "Crosses_Cutoff"):
        part = raw if sample == "All" else raw.loc[raw.Sample.eq(sample)]
        grid = pd.MultiIndex.from_product([range(1, 5), range(1, 5)], names=["From_State", "To_State"])
        counts = part.groupby(["From_State", "To_State"]).size().reindex(
            grid, fill_value=0).rename("Count").reset_index()
        totals = counts.groupby("From_State")["Count"].transform("sum")
        counts["Empirical_Probability"] = counts["Count"].div(totals.where(totals.ne(0)))
        counts.insert(0, "Sample", sample); counts["From_State_Label"] = "Ordered_State_" + counts.From_State.astype(str)
        counts["To_State_Label"] = "Ordered_State_" + counts.To_State.astype(str); outputs.append(counts)
    return pd.concat(outputs, ignore_index=True)


def filtered_episodes(frame: pd.DataFrame, cutoff="2017-12-31") -> pd.DataFrame:
    dates, cutoff = pd.to_datetime(frame["Date"]), pd.Timestamp(cutoff)
    valid = frame["HMM_Eligible"].fillna(False) & frame["HMM_Filtered_State"].notna(); episodes, start = [], None
    for i in range(len(frame) + 1):
        boundary = i == len(frame) or not valid.iloc[i] or (start is not None and int(frame.HMM_Filtered_State.iloc[i]) != int(frame.HMM_Filtered_State.iloc[start]))
        if start is not None and boundary:
            end = i - 1; sample = "Train" if dates.iloc[end] <= cutoff else ("Test" if dates.iloc[start] > cutoff else "Crosses_Cutoff")
            episodes.append({"State": int(frame.HMM_Filtered_State.iloc[start]), "Start_Date": dates.iloc[start],
                "End_Date": dates.iloc[end], "Length": end - start + 1, "Sample": sample}); start = None
        if i < len(frame) and valid.iloc[i] and start is None: start = i
    return pd.DataFrame(episodes)


def filtered_episode_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in ("All", "Train", "Test", "Crosses_Cutoff"):
        source = episodes if sample == "All" else episodes.loc[episodes.Sample.eq(sample)]
        for state in range(1, 5):
            values = source.loc[source.State.eq(state), "Length"]
            rows.append({"Sample": sample, "Ordered_State": state, "Episode_Count": len(values),
                "Mean_Length": values.mean(), "Median_Length": values.median(), "Std_Length": values.std(ddof=1),
                "Minimum_Length": values.min(), "Maximum_Length": values.max(), "Q25_Length": values.quantile(.25),
                "Q75_Length": values.quantile(.75), "Q90_Length": values.quantile(.9)})
    return pd.DataFrame(rows)


def transition_confidence_summary(frame: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for h in horizons:
        modeled = frame[f"Transition_Sample_{h}D"].isin(["Train", "Test"])
        high = frame[f"Transition_Window_All_High_Confidence_{h}D"].eq(True)
        for target in target_columns(h):
            for sample in ("All", "Train", "Test"):
                mask = modeled if sample == "All" else frame[f"Transition_Sample_{h}D"].eq(sample)
                values = frame.loc[mask, target].dropna(); high_values = frame.loc[mask & high, target].dropna()
                rows.append({"horizon": h, "target": target, "sample": sample,
                    "all_available_count": len(values), "all_event_rate": values.mean(),
                    "all_high_confidence_count": len(high_values), "all_high_confidence_event_rate": high_values.mean(),
                    "excluded_by_high_confidence_restriction": len(values) - len(high_values)})
    return pd.DataFrame(rows)
