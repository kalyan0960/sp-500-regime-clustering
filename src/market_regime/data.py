"""Raw SPY and VIX acquisition, normalization, validation, and checkpoint utilities."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


class DownloadError(RuntimeError):
    """Raised when Yahoo Finance does not provide usable data for a ticker."""


class DataValidationError(ValueError):
    """Raised when a raw market dataset fails a documented quality check."""


_PRICE_COLUMNS = ("Open", "High", "Low", "Close")
_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj_Close",
    "adj_close": "Adj_Close",
    "adjclose": "Adj_Close",
    "volume": "Volume",
}


def download_yahoo_ticker(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Download one ticker through an inclusive research end date from Yahoo Finance.

    Yahoo Finance treats ``end`` as exclusive. The request therefore uses one
    calendar day after ``end_date`` and explicitly filters the returned dates
    back to the configured inclusive interval. ``auto_adjust=False`` preserves
    raw OHLC columns and any Adjusted Close column returned by Yahoo Finance.
    """
    start = pd.Timestamp(start_date).date()
    inclusive_end = pd.Timestamp(end_date).date()
    if start > inclusive_end:
        raise ValueError("start_date must be on or before end_date.")

    try:
        downloaded = yf.download(
            tickers=ticker,
            start=start.isoformat(),
            end=(inclusive_end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="column",
            threads=False,
            multi_level_index=False,
            timeout=30,
        )
    except (ConnectionError, OSError, TimeoutError) as error:
        raise DownloadError(f"Yahoo Finance download failed for {ticker}.") from error

    if downloaded.empty:
        raise DownloadError(f"Yahoo Finance returned no data for {ticker}.")

    return filter_configured_dates(downloaded, start, inclusive_end)


def flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten ordinary or yfinance MultiIndex columns without discarding labels."""
    flattened = frame.copy()
    if not isinstance(flattened.columns, pd.MultiIndex):
        return flattened

    ticker_upper = ticker.upper()
    names: list[str] = []
    for column in flattened.columns.to_flat_index():
        labels = [str(label).strip() for label in column if str(label).strip()]
        recognized = next(
            (label for label in labels if _canonical_column_name(label) is not None),
            None,
        )
        if recognized is not None:
            names.append(recognized)
            continue

        non_ticker_labels = [label for label in labels if label.upper() != ticker_upper]
        if len(non_ticker_labels) == 1:
            names.append(non_ticker_labels[0])
            continue
        raise DataValidationError(
            f"Cannot safely flatten yfinance column {column!r} for ticker {ticker}."
        )

    if len(names) != len(set(names)):
        raise DataValidationError("Flattening yfinance columns would create duplicate names.")
    flattened.columns = names
    return flattened


def standardize_column_names(frame: pd.DataFrame, *, vix_prefix: bool = False) -> pd.DataFrame:
    """Apply clear, canonical raw-market column names while retaining all columns."""
    standardized = frame.copy()
    renamed: dict[Any, str] = {}
    for column in standardized.columns:
        canonical = _canonical_column_name(str(column))
        if canonical is None:
            cleaned = str(column).strip().replace(" ", "_")
            canonical = cleaned
        renamed[column] = f"VIX_{canonical}" if vix_prefix else canonical

    if len(set(renamed.values())) != len(renamed):
        raise DataValidationError("Column standardization would create duplicate names.")
    return standardized.rename(columns=renamed)


def normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert to a timezone-naive, sorted, uniquely named daily Date index."""
    normalized = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(normalized.index, errors="raise"))
    except (TypeError, ValueError) as error:
        raise DataValidationError("Raw data index cannot be converted to dates.") from error

    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    normalized.attrs["unsorted_dates_detected"] = not index.is_monotonic_increasing
    normalized.index = index
    normalized.index.name = "Date"
    return normalized.sort_index()


def filter_configured_dates(
    frame: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Restrict raw observations to the configured inclusive research dates."""
    normalized = normalize_date_index(frame)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    return normalized.loc[(normalized.index >= start) & (normalized.index <= end)].copy()


def prepare_raw_ticker_data(
    frame: pd.DataFrame,
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    *,
    vix_prefix: bool = False,
) -> pd.DataFrame:
    """Flatten, name, normalize, and date-filter one downloaded ticker dataset."""
    flattened = flatten_yfinance_columns(frame, ticker)
    standardized = standardize_column_names(flattened, vix_prefix=vix_prefix)
    return filter_configured_dates(standardized, start_date, end_date)


def validate_spy_data(
    frame: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
) -> dict[str, Any]:
    """Validate raw SPY OHLC and volume data and return its quality report."""
    return _validate_raw_data(
        frame,
        required_columns=(*_PRICE_COLUMNS, "Volume"),
        price_columns=_PRICE_COLUMNS + (("Adj_Close",) if "Adj_Close" in frame.columns else ()),
        start_date=start_date,
        end_date=end_date,
        volume_column="Volume",
        dataset_name="SPY",
    )


def validate_vix_data(
    frame: pd.DataFrame,
    start_date: str | date,
    end_date: str | date,
) -> dict[str, Any]:
    """Validate raw VIX price data and return its quality report."""
    vix_prices = tuple(f"VIX_{name}" for name in _PRICE_COLUMNS)
    return _validate_raw_data(
        frame,
        required_columns=vix_prices,
        price_columns=vix_prices,
        start_date=start_date,
        end_date=end_date,
        volume_column=None,
        dataset_name="VIX",
    )


def compare_trading_calendars(spy: pd.DataFrame, vix: pd.DataFrame) -> dict[str, Any]:
    """Report shared and ticker-specific trading dates before date alignment."""
    spy_dates = pd.DatetimeIndex(spy.index).sort_values()
    vix_dates = pd.DatetimeIndex(vix.index).sort_values()
    common_dates = spy_dates.intersection(vix_dates)
    spy_only = spy_dates.difference(vix_dates)
    vix_only = vix_dates.difference(spy_dates)
    return {
        "common_dates": common_dates,
        "spy_only_dates": spy_only,
        "vix_only_dates": vix_only,
        "common_count": len(common_dates),
        "spy_only_count": len(spy_only),
        "vix_only_count": len(vix_only),
        "spy_only_examples": [timestamp.date().isoformat() for timestamp in spy_only[:5]],
        "vix_only_examples": [timestamp.date().isoformat() for timestamp in vix_only[:5]],
    }


def create_aligned_raw_market_data(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Inner-join validated raw SPY and VIX data on common trading dates only."""
    overlap = set(spy.columns).intersection(vix.columns)
    if overlap:
        raise DataValidationError(
            f"SPY and VIX columns would collide during alignment: {sorted(overlap)}"
        )
    aligned = spy.join(vix, how="inner", validate="one_to_one").sort_index()
    aligned.index.name = "Date"
    if aligned.empty:
        raise DataValidationError("SPY and VIX have no common trading dates.")
    return aligned


def save_csv_checkpoint(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save a raw-data checkpoint with an explicit Date column and full precision."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_save = frame.copy()
    to_save.index.name = "Date"
    to_save.to_csv(destination, index=True, index_label="Date")
    return destination


def _canonical_column_name(name: str) -> str | None:
    """Return a canonical raw-market name when a Yahoo Finance label is known."""
    normalized = " ".join(name.strip().lower().replace("_", " ").split())
    return _COLUMN_MAP.get(normalized)


def _validate_raw_data(
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    price_columns: tuple[str, ...],
    start_date: str | date,
    end_date: str | date,
    volume_column: str | None,
    dataset_name: str,
) -> dict[str, Any]:
    """Build a validation report and reject raw data that violates required rules."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{dataset_name} index must be a DatetimeIndex.")

    missing_required = [column for column in required_columns if column not in frame.columns]
    numeric_columns = [column for column in required_columns if column in frame.columns]
    outside_dates = (
        (frame.index < pd.Timestamp(start_date)) | (frame.index > pd.Timestamp(end_date))
    ).sum()
    report: dict[str, Any] = {
        "first_available_date": frame.index.min().date().isoformat() if not frame.empty else None,
        "last_available_date": frame.index.max().date().isoformat() if not frame.empty else None,
        "row_count": len(frame),
        "missing_required_columns": missing_required,
        "numeric_columns": {
            column: pd.api.types.is_numeric_dtype(frame[column]) for column in numeric_columns
        },
        "missing_values": {column: int(frame[column].isna().sum()) for column in frame.columns},
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "unsorted_dates": int(
            bool(frame.attrs.get("unsorted_dates_detected", False))
            or not frame.index.is_monotonic_increasing
        ),
        "dates_outside_configured_range": int(outside_dates),
        "nonfinite_values": {},
        "nonpositive_prices": {},
    }

    for column in numeric_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].to_numpy(dtype=float, copy=False)
            report["nonfinite_values"][column] = int((~np.isfinite(values)).sum())
        else:
            report["nonfinite_values"][column] = None

    for column in price_columns:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            report["nonpositive_prices"][column] = int((frame[column] <= 0).sum())

    if volume_column is not None and volume_column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[volume_column]):
            report["zero_volume_observations"] = int((frame[volume_column] == 0).sum())
            report["negative_volume_observations"] = int((frame[volume_column] < 0).sum())
        else:
            report["zero_volume_observations"] = None
            report["negative_volume_observations"] = None

    failures: list[str] = []
    if frame.empty:
        failures.append("dataset is empty")
    if missing_required:
        failures.append(f"missing required columns: {missing_required}")
    if not all(report["numeric_columns"].values()):
        failures.append("required columns must have numeric types")
    if any(report["missing_values"].values()):
        failures.append("missing values are present")
    if report["duplicate_dates"]:
        failures.append("duplicate dates are present")
    if report["unsorted_dates"]:
        failures.append("dates were not supplied in ascending order")
    if report["dates_outside_configured_range"]:
        failures.append("dates fall outside the configured range")
    if any(value not in (0, None) for value in report["nonfinite_values"].values()):
        failures.append("non-finite numeric values are present")
    if any(value != 0 for value in report["nonpositive_prices"].values()):
        failures.append("nonpositive prices are present")
    if volume_column is not None:
        if report.get("zero_volume_observations") != 0:
            failures.append("zero-volume observations are present")
        if report.get("negative_volume_observations") != 0:
            failures.append("negative-volume observations are present")

    if failures:
        raise DataValidationError(f"{dataset_name} validation failed: {'; '.join(failures)}")
    return report
