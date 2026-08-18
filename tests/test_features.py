"""Deterministic tests for leakage-safe Stage 7 features."""

import numpy as np
import pandas as pd
import pytest

from market_regime.features import (
    FeatureValidationError, assign_chronological_samples, build_pre_garch_features,
    calculate_abnormal_volume, calculate_previous_volume_benchmark, calculate_returns,
    calculate_rolling_drawdown, save_feature_checkpoint,
)


def _frame(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2017-01-02", periods=periods, freq="B")
    price = pd.Series(np.arange(100, 100 + periods, dtype=float), index=index)
    return pd.DataFrame({"Adj_Close": price, "Open": price - 0.5, "Volume": 1000.0, "VIX_Close": 20.0}, index=index)


def test_returns_use_prior_close_not_open_and_log_identity() -> None:
    frame = _frame(3)
    frame["Open"] = [1.0, 999.0, 1.0]
    returns = calculate_returns(frame["Adj_Close"])
    assert returns.iloc[1]["Simple_Return"] == pytest.approx(0.01)
    assert returns.iloc[1]["Log_Return"] == pytest.approx(np.log(1.01))
    assert np.allclose(returns["Log_Return"].iloc[1:], np.log1p(returns["Simple_Return"].iloc[1:]))


def test_nonpositive_price_and_volume_are_rejected() -> None:
    with pytest.raises(FeatureValidationError): calculate_returns(pd.Series([1.0, 0.0]))
    with pytest.raises(FeatureValidationError): calculate_previous_volume_benchmark(pd.Series([1.0, -1.0]))


def test_previous_volume_benchmark_excludes_today() -> None:
    volume = pd.Series([100.0] * 20 + [10000.0])
    benchmark = calculate_previous_volume_benchmark(volume, 20)
    assert benchmark.iloc[:20].isna().all()
    assert benchmark.iloc[20] == pytest.approx(100.0)
    abnormal = calculate_abnormal_volume(volume, benchmark)
    assert abnormal.iloc[20]["Abnormal_Volume"] == pytest.approx(100.0)
    assert abnormal.iloc[20]["Log_Abnormal_Volume"] == pytest.approx(np.log(100.0))


def test_drawdown_uses_full_windows_and_never_positive() -> None:
    price = pd.Series([100.0, 110.0, 105.0, 120.0])
    drawdown = calculate_rolling_drawdown(price, 2)
    assert pd.isna(drawdown.iloc[0]["Drawdown_2"])
    assert drawdown.iloc[1]["Drawdown_2"] == pytest.approx(0.0)
    assert drawdown.iloc[2]["Drawdown_2"] == pytest.approx(105 / 110 - 1)
    assert drawdown["Drawdown_2"].dropna().le(1e-12).all()


def test_chronological_labels_are_not_random() -> None:
    index = pd.DatetimeIndex(["2017-12-29", "2018-01-02"])
    labels = assign_chronological_samples(index, "2017-12-31", "2018-01-01")
    assert labels["Sample"].tolist() == ["Train", "Test"]
    assert labels["Is_Train"].tolist() == [True, False]


def test_build_features_preserves_rows_input_and_no_look_ahead(tmp_path) -> None:
    frame = _frame(270)
    original = frame.copy(deep=True)
    kwargs = dict(volume_window=20, primary_drawdown_window=252, robustness_drawdown_window=60, training_end="2017-12-31", test_start="2018-01-01")
    prefix = build_pre_garch_features(frame.iloc[:260], **kwargs)
    full = build_pre_garch_features(frame, **kwargs)
    columns = ["Simple_Return", "Log_Return", "Abnormal_Volume", "Drawdown_252"]
    pd.testing.assert_frame_equal(prefix[columns], full.loc[prefix.index, columns])
    pd.testing.assert_frame_equal(frame, original)
    assert len(full) == len(frame)
    path = save_feature_checkpoint(full, tmp_path / "features.csv")
    reloaded = pd.read_csv(path, index_col="Date", parse_dates=True)
    assert len(reloaded) == len(full)
    assert reloaded.columns.tolist() == full.columns.tolist()
