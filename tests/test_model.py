"""Tests for the regression pipeline (src/model.py), metrics
(src/evaluate.py) and model selection/saving (src/train.py)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import clean_data, load_raw_data, split_data
from src.evaluate import interpret_coefficients, regression_metrics
from src.model import VARIANTS, build_pipeline, coefficient_table, input_columns
from src.train import MODEL_PATH, select_variant


@pytest.fixture
def synthetic_cars() -> tuple[pd.DataFrame, pd.Series]:
    """60 cars whose price follows a known log-linear rule plus noise."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "Year": rng.integers(2005, 2019, n),
        "Present_Price": rng.uniform(3, 30, n),
        "Kms_Driven": rng.integers(5_000, 150_000, n),
        "Fuel_Type": rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.6, 0.3, 0.1]),
        "Transmission": rng.choice(["Manual", "Automatic"], n, p=[0.8, 0.2]),
    })
    age = config.REFERENCE_YEAR - X["Year"]
    y = np.exp(0.9 * np.log(X["Present_Price"]) - 0.1 * age
               + 0.15 * (X["Fuel_Type"] == "Diesel") + rng.normal(0, 0.05, n))
    return X, pd.Series(y, name="Selling_Price")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_trains_and_predicts(variant, synthetic_cars):
    X, y = synthetic_cars
    model = build_pipeline(variant).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_log_target_predictions_are_always_positive(synthetic_cars):
    X, y = synthetic_cars
    model = build_pipeline("log_log").fit(X, y)
    extreme = X.head(1).assign(Year=2003, Kms_Driven=500_000, Present_Price=2.0)
    assert (model.predict(extreme) > 0).all()


def test_pipeline_features_and_reference_levels(synthetic_cars):
    X, y = synthetic_cars
    terms = coefficient_table(build_pipeline("log_log").fit(X, y))["term"].tolist()
    assert terms == ["(intercept)", "Car_Age", "Kms_Driven", "log_Present_Price",
                     "Fuel_Type_CNG", "Fuel_Type_Diesel", "Transmission_Automatic"]
    # Year itself never reaches the regression, only the derived age.
    assert "Year" not in terms


def test_pipeline_recovers_known_coefficients(synthetic_cars):
    X, y = synthetic_cars
    coefs = coefficient_table(build_pipeline("log_log").fit(X, y)).set_index("term")
    assert coefs.loc["log_Present_Price", "coefficient"] == pytest.approx(0.9, abs=0.05)
    assert coefs.loc["Car_Age", "coefficient"] == pytest.approx(-0.1, abs=0.02)
    assert coefs.loc["Fuel_Type_Diesel", "coefficient"] == pytest.approx(0.15, abs=0.05)


def test_without_present_price_schema(synthetic_cars):
    X, y = synthetic_cars
    cols = input_columns(include_present_price=False)
    assert "Present_Price" not in cols
    terms = coefficient_table(build_pipeline("log_target", include_present_price=False)
                              .fit(X[cols], y))["term"]
    assert not terms.str.contains("Present_Price").any()


def test_missing_input_column_raises(synthetic_cars):
    X, y = synthetic_cars
    model = build_pipeline("log_log").fit(X, y)
    with pytest.raises((KeyError, ValueError)):
        model.predict(X.drop(columns=["Kms_Driven"]))


def test_saved_pipeline_gives_identical_predictions(synthetic_cars, tmp_path):
    X, y = synthetic_cars
    model = build_pipeline("log_log").fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X), model.predict(X))


def test_regression_metrics_known_values():
    m = regression_metrics(pd.Series([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["MAE"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["RMSE"] == pytest.approx(np.sqrt(4 / 3), abs=1e-3)
    assert m["R2"] == pytest.approx(1 - 4 / 2, abs=1e-3)
    assert m["n"] == 3


def test_select_variant_picks_lowest_rmse():
    cv = {"a": {"RMSE_mean": 2.0}, "b": {"RMSE_mean": 0.9}, "c": {"RMSE_mean": 1.5}}
    assert select_variant(cv) == "b"


def test_interpretation_converts_log_coefficients_to_percent(synthetic_cars):
    X, y = synthetic_cars
    model = build_pipeline("log_log").fit(X, y)
    table = interpret_coefficients(model, log_target=True).set_index("term")
    b = table.loc["Car_Age", "coefficient"]
    assert f"{(np.exp(b) - 1) * 100:+.1f}%" in table.loc["Car_Age", "effect"]
    assert "vs Petrol" in table.loc["Fuel_Type_Diesel", "effect"]


# --- Tests against the real dataset / saved model -----------------------

real_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(), reason="raw dataset not downloaded"
)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
@real_data
def test_saved_model_bundle_matches_training_schema():
    bundle = joblib.load(MODEL_PATH)
    assert bundle["input_columns"] == input_columns()
    assert bundle["variant"] in VARIANTS
    assert bundle["reference_year"] == config.REFERENCE_YEAR

    train, test = split_data(clean_data(load_raw_data())[0])
    refit = build_pipeline(bundle["variant"]).fit(
        train[input_columns()], train[config.TARGET])
    np.testing.assert_allclose(bundle["model"].predict(test[input_columns()]),
                               refit.predict(test[input_columns()]))
