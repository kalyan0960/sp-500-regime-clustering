"""Deterministic tests for Stage 6 raw-data utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime.data import (
    DataValidationError,
    compare_trading_calendars,
    create_aligned_raw_market_data,
    filter_configured_dates,
    flatten_yfinance_columns,
    normalize_date_index,
    prepare_raw_ticker_data,
    save_csv_checkpoint,
    validate_spy_data,
    validate_vix_data,
)


START = "2020-01-01"
END = "2020-01-10"


def _spy_frame(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    index = index if index is not None else pd.date_range("2020-01-02", periods=3, freq="B")
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "Open": 100.0 + values,
            "High": 101.0 + values,
            "Low": 99.0 + values,
            "Close": 100.5 + values,
            "Adj_Close": 100.4 + values,
            "Volume": 1000 + (100 * values),
        },
        index=index,
    )


def _vix_frame(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    index = index if index is not None else pd.date_range("2020-01-02", periods=3, freq="B")
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "VIX_Open": 15.0 + values,
            "VIX_High": 16.0 + values,
            "VIX_Low": 14.0 + values,
            "VIX_Close": 15.5 + values,
        },
        index=index,
    )


def test_flatten_yfinance_multiindex_columns() -> None:
    frame = _spy_frame().drop(columns="Adj_Close")
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    flattened = flatten_yfinance_columns(frame, "SPY")
    assert list(flattened.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_prepare_ordinary_columns_and_filter_dates() -> None:
    frame = _spy_frame(pd.date_range("2019-12-31", periods=3, freq="B"))
    prepared = prepare_raw_ticker_data(frame, "SPY", START, END)
    assert list(prepared.columns) == ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    assert prepared.index.name == "Date"
    assert prepared.index.min() >= pd.Timestamp(START)


def test_normalize_datetime_timezone_and_sorting() -> None:
    index = pd.DatetimeIndex(["2020-01-03 13:00:00+00:00", "2020-01-02 13:00:00+00:00"])
    normalized = normalize_date_index(_spy_frame(index))
    assert normalized.index.tz is None
    assert normalized.index.is_monotonic_increasing
    assert normalized.attrs["unsorted_dates_detected"] is True


def test_duplicate_date_detection() -> None:
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-03"])
    with pytest.raises(DataValidationError, match="duplicate dates"):
        validate_spy_data(_spy_frame(index), START, END)


def test_missing_required_spy_column_detection() -> None:
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_spy_data(_spy_frame().drop(columns="Close"), START, END)


@pytest.mark.parametrize("volume", [0, -1])
def test_nonpositive_spy_volume_detection(volume: int) -> None:
    frame = _spy_frame()
    frame.loc[frame.index[0], "Volume"] = volume
    with pytest.raises(DataValidationError, match="volume"):
        validate_spy_data(frame, START, END)


def test_nonpositive_price_detection() -> None:
    frame = _spy_frame()
    frame.loc[frame.index[0], "Close"] = 0
    with pytest.raises(DataValidationError, match="nonpositive prices"):
        validate_spy_data(frame, START, END)


def test_filter_configured_dates() -> None:
    frame = _spy_frame(pd.date_range("2019-12-31", periods=5, freq="B"))
    filtered = filter_configured_dates(frame, "2020-01-02", "2020-01-03")
    assert filtered.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


def test_vix_required_columns_are_validated() -> None:
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_vix_data(_vix_frame().drop(columns="VIX_Close"), START, END)


def test_alignment_prevents_column_collisions() -> None:
    vix = _vix_frame().rename(columns={"VIX_Close": "Close"})
    with pytest.raises(DataValidationError, match="collide"):
        create_aligned_raw_market_data(_spy_frame(), vix)


def test_common_date_alignment_and_calendar_report() -> None:
    spy = _spy_frame(pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"]))
    vix = _vix_frame(pd.DatetimeIndex(["2020-01-02", "2020-01-06", "2020-01-07"]))
    report = compare_trading_calendars(spy, vix)
    aligned = create_aligned_raw_market_data(spy, vix)
    assert report["common_count"] == 2
    assert report["spy_only_count"] == 1
    assert report["vix_only_count"] == 1
    assert aligned.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]


def test_alignment_does_not_create_weekend_rows() -> None:
    spy = _spy_frame(pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"]))
    vix = _vix_frame(pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"]))
    aligned = create_aligned_raw_market_data(spy, vix)
    assert all(timestamp.weekday() < 5 for timestamp in aligned.index)
    assert pd.Timestamp("2020-01-04") not in aligned.index


def test_csv_save_and_reload_consistency(tmp_path: Path) -> None:
    frame = _spy_frame()
    checkpoint = save_csv_checkpoint(frame, tmp_path / "spy.csv")
    reloaded = pd.read_csv(checkpoint, index_col="Date", parse_dates=True)
    assert reloaded.index.name == "Date"
    assert reloaded.index.tolist() == frame.index.tolist()
    assert reloaded.columns.tolist() == frame.columns.tolist()
    assert len(reloaded) == len(frame)
    assert np.allclose(reloaded["Close"], frame["Close"])
