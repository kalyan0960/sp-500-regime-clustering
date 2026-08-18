"""Leakage-safe pre-GARCH market feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class FeatureValidationError(ValueError):
    """Raised when a source series cannot support the documented features."""


def select_adjusted_price(frame: pd.DataFrame) -> pd.Series:
    """Return valid SPY Adjusted Close; no undocumented Close fallback is allowed."""
    if "Adj_Close" not in frame.columns:
        raise FeatureValidationError("Adj_Close is unavailable; price selection requires a documented decision.")
    return validate_positive_price(frame["Adj_Close"], name="Adj_Close")


def validate_positive_price(price: pd.Series, *, name: str = "price") -> pd.Series:
    """Validate a numeric, finite, strictly positive price series without mutation."""
    values = pd.to_numeric(price, errors="raise").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise FeatureValidationError(f"{name} must be finite and strictly positive.")
    return values.copy()


def calculate_returns(price: pd.Series) -> pd.DataFrame:
    """Calculate close-to-close simple and natural-log returns with no filling."""
    validated = validate_positive_price(price)
    simple = validated.pct_change(fill_method=None).rename("Simple_Return")
    log_return = np.log(validated / validated.shift(1)).rename("Log_Return")
    return pd.concat([simple, log_return], axis=1)


def calculate_previous_volume_benchmark(volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the prior-only rolling volume mean, excluding the current day."""
    if window <= 0:
        raise ValueError("window must be positive.")
    validated = _validate_positive_volume(volume)
    return validated.shift(1).rolling(window=window, min_periods=window).mean().rename(
        f"Volume_MA{window}_Previous"
    )


def calculate_abnormal_volume(volume: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Calculate positive abnormal volume and its natural logarithm where defined."""
    validated_volume = _validate_positive_volume(volume)
    if not benchmark.index.equals(validated_volume.index):
        raise FeatureValidationError("Volume and benchmark indexes must match.")
    valid_benchmark = benchmark.dropna()
    if not valid_benchmark.empty and ((valid_benchmark <= 0).any() or not np.isfinite(valid_benchmark).all()):
        raise FeatureValidationError("Defined volume benchmarks must be finite and positive.")
    abnormal = (validated_volume / benchmark).rename("Abnormal_Volume")
    if not abnormal.dropna().gt(0).all():
        raise FeatureValidationError("Defined abnormal volume must be positive.")
    return pd.concat([abnormal, np.log(abnormal).rename("Log_Abnormal_Volume")], axis=1)


def calculate_rolling_drawdown(price: pd.Series, window: int) -> pd.DataFrame:
    """Calculate full-window rolling peaks and drawdown using observations through today."""
    if window <= 0:
        raise ValueError("window must be positive.")
    validated = validate_positive_price(price)
    peak = validated.rolling(window=window, min_periods=window).max().rename(f"Rolling_Peak_{window}")
    drawdown = (validated / peak - 1).rename(f"Drawdown_{window}")
    if (drawdown.dropna() > 1e-12).any():
        raise FeatureValidationError("Drawdown cannot be materially positive.")
    return pd.concat([peak, drawdown], axis=1)


def assign_chronological_samples(
    index: pd.DatetimeIndex, training_end: str, test_start: str
) -> pd.DataFrame:
    """Assign deterministic Train/Test labels from non-overlapping configured dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    train_end = pd.Timestamp(training_end)
    test_begin = pd.Timestamp(test_start)
    if train_end >= test_begin:
        raise FeatureValidationError("training_end must be before test_start.")
    sample = pd.Series(index=dates, dtype="object", name="Sample")
    sample.loc[dates <= train_end] = "Train"
    sample.loc[dates >= test_begin] = "Test"
    if sample.isna().any():
        raise FeatureValidationError("Configured sample boundaries leave dates unassigned.")
    return pd.DataFrame({"Sample": sample, "Is_Train": sample.eq("Train")}, index=dates)


def build_pre_garch_features(
    frame: pd.DataFrame,
    *,
    volume_window: int,
    primary_drawdown_window: int,
    robustness_drawdown_window: int,
    training_end: str,
    test_start: str,
) -> pd.DataFrame:
    """Build the complete dated Stage 7 feature frame without dropping source rows."""
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise FeatureValidationError("Input data must have a unique DatetimeIndex.")
    if not frame.index.is_monotonic_increasing:
        raise FeatureValidationError("Input dates must be sorted ascending.")
    if "Volume" not in frame.columns or "VIX_Close" not in frame.columns:
        raise FeatureValidationError("Input data must contain Volume and VIX_Close.")

    result = frame.copy()
    price = select_adjusted_price(result)
    result["Price_Used"] = price
    returns = calculate_returns(price)
    benchmark = calculate_previous_volume_benchmark(result["Volume"], volume_window)
    abnormal = calculate_abnormal_volume(result["Volume"], benchmark)
    primary_drawdown = calculate_rolling_drawdown(price, primary_drawdown_window)
    robustness_drawdown = calculate_rolling_drawdown(price, robustness_drawdown_window)
    samples = assign_chronological_samples(result.index, training_end, test_start)

    result = result.join(returns).join(benchmark).join(abnormal)
    result = result.join(primary_drawdown).join(robustness_drawdown).join(samples)
    result.index.name = "Date"
    return result


def save_feature_checkpoint(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save the pre-GARCH feature frame with its Date index explicitly retained."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=True, index_label="Date")
    return destination


def _validate_positive_volume(volume: pd.Series) -> pd.Series:
    """Validate a numeric, finite, strictly positive volume series."""
    values = pd.to_numeric(volume, errors="raise").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise FeatureValidationError("Volume must be finite and strictly positive.")
    return values.copy()
