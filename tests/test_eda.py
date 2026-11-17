"""Tests for the Stage 2 helpers: car age, the train/test split and the
EDA diagnostics. The last tests use the real CSV and are skipped if it is
absent."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import add_car_age, clean_data, load_raw_data, split_data
from src.eda import (
    cooks_distance,
    rows_outside_training_range,
    run_eda,
    variance_inflation_factors,
)


def test_add_car_age_uses_reference_year_and_keeps_input():
    df = pd.DataFrame({"Year": [2018, 2003]})
    out = add_car_age(df)
    assert out["Car_Age"].tolist() == [config.REFERENCE_YEAR - 2018,
                                       config.REFERENCE_YEAR - 2003]
    assert "Car_Age" not in df.columns


def test_split_is_reproducible_disjoint_and_complete():
    df = pd.DataFrame({"x": range(100)})
    train1, test1 = split_data(df)
    train2, test2 = split_data(df)
    assert train1.index.equals(train2.index) and test1.index.equals(test2.index)
    assert set(train1.index).isdisjoint(test1.index)
    assert len(train1) + len(test1) == 100
    assert len(test1) == 20


def test_vif_detects_collinear_column():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({
        "a": a,
        "a_copy": a + rng.normal(scale=0.05, size=200),  # nearly identical
        "independent": rng.normal(size=200),
    })
    vif = variance_inflation_factors(X)
    assert vif["a"] > 50 and vif["a_copy"] > 50
    assert vif["independent"] < 1.5


def test_cooks_distance_flags_planted_outlier():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, size=50)
    y = 2 * x + 1 + rng.normal(scale=0.5, size=50)
    x[0], y[0] = 20.0, -10.0           # high leverage AND far off the line
    d = cooks_distance(pd.DataFrame({"x": x}), pd.Series(y))
    assert d.argmax() == 0
    assert d[0] > 4 / len(x)
    assert (d >= 0).all()


def test_rows_outside_training_range():
    train = pd.DataFrame({"p": [1.0, 5.0, 10.0]})
    test = pd.DataFrame({"p": [3.0, 12.0, 0.5]})
    assert rows_outside_training_range(train, test, ["p"])["p"].tolist() == [12.0, 0.5]


# --- Tests against the real dataset -------------------------------------

real_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(), reason="raw dataset not downloaded"
)


@real_data
def test_real_split_sizes():
    train, test = split_data(clean_data(load_raw_data())[0])
    assert (len(train), len(test)) == (158, 40)


@real_data
def test_run_eda_writes_all_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    summary = run_eda()
    assert summary["rows_train"] == 158
    assert len(summary["figures"]) == 8
    for name in summary["figures"]:
        assert (tmp_path / name).stat().st_size > 10_000
    # The land cruiser is the only test car outside the training ranges.
    outside = summary["test_rows_outside_training_feature_range"]
    assert [r["Car_Name"] for r in outside] == ["land cruiser"]
