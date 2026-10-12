"""Tests for src/predict.py and the Streamlit app (app.py).

The prediction tests build a small model in a temporary folder, so they do
not need the real dataset. The app tests need the trained model file and
are skipped without it.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.model import build_pipeline, input_columns
from src.predict import (
    ModelNotFoundError,
    load_model_bundle,
    predict_price,
    validate_inputs,
)
from src.train import MODEL_PATH

VALID = {"Year": 2015, "Present_Price": 8.0, "Kms_Driven": 40_000,
         "Fuel_Type": "Petrol", "Transmission": "Manual"}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> dict:
    """A bundle with the same structure as the one src.train saves."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
        "Year": rng.integers(2005, 2019, n),
        "Present_Price": rng.uniform(3, 30, n),
        "Kms_Driven": rng.integers(5_000, 150_000, n),
        "Fuel_Type": rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.6, 0.3, 0.1]),
        "Transmission": rng.choice(["Manual", "Automatic"], n),
    })
    y = np.exp(0.8 * np.log(X["Present_Price"]) - 0.1 * (2020 - X["Year"]))
    saved = {
        "model": build_pipeline("log_log").fit(X, y),
        "input_columns": input_columns(),
        "variant": "log_log",
        "reference_year": config.REFERENCE_YEAR,
        "training_ranges": {"Year": [2005.0, 2018.0], "Present_Price": [3.0, 30.0],
                            "Kms_Driven": [5000.0, 150000.0]},
        "categories": {"Fuel_Type": ["CNG", "Diesel", "Petrol"],
                       "Transmission": ["Automatic", "Manual"]},
    }
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(saved, path)
    return load_model_bundle(path)


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_model_bundle(tmp_path / "nope.joblib")


def test_incomplete_model_file_is_rejected(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": None}, path)
    with pytest.raises(ValueError, match="retrain"):
        load_model_bundle(path)


def test_valid_input_gives_positive_price_without_warnings(bundle):
    result = predict_price(bundle, VALID)
    assert result.errors == [] and result.warnings == []
    assert 0 < result.price < VALID["Present_Price"]


def test_older_car_is_cheaper(bundle):
    new = predict_price(bundle, {**VALID, "Year": 2017}).price
    old = predict_price(bundle, {**VALID, "Year": 2008}).price
    assert old < new


@pytest.mark.parametrize("change, message", [
    ({"Year": 2030}, "Year must be between"),
    ({"Year": 2015.5}, "whole number"),
    ({"Present_Price": 0}, "Present price"),
    ({"Present_Price": -3}, "Present price"),
    ({"Present_Price": "abc"}, "Present price"),
    ({"Present_Price": float("nan")}, "Present price"),
    ({"Kms_Driven": -10}, "Kilometres"),
    ({"Fuel_Type": "Electric"}, "Fuel_Type must be one of"),
    ({"Transmission": None}, "Transmission is required"),
])
def test_invalid_inputs_are_rejected(bundle, change, message):
    result = predict_price(bundle, {**VALID, **change})
    assert result.price is None
    assert any(message in e for e in result.errors)


def test_missing_field_is_reported(bundle):
    inputs = {k: v for k, v in VALID.items() if k != "Kms_Driven"}
    errors, _ = validate_inputs(bundle, inputs)
    assert errors == ["Kms_Driven is required."]


def test_out_of_range_inputs_warn_but_still_predict(bundle):
    result = predict_price(bundle, {**VALID, "Present_Price": 90.0})
    assert result.price is not None
    assert any("outside the training range" in w for w in result.warnings)


def test_cng_warns_about_small_sample(bundle):
    result = predict_price(bundle, {**VALID, "Fuel_Type": "CNG"})
    assert any("CNG" in w for w in result.warnings)


def test_string_numbers_from_forms_are_accepted(bundle):
    as_text = {**VALID, "Year": "2015", "Present_Price": "8.0", "Kms_Driven": "40000"}
    assert predict_price(bundle, as_text).price == pytest.approx(
        predict_price(bundle, VALID).price)


# --- Streamlit app ------------------------------------------------------

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run `python -m src.train` first")


@needs_model
def test_app_starts_and_shows_test_metrics():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py")).run(timeout=60)
    assert not at.exception
    assert at.title[0].value.startswith("Indian Used Car")
    assert [m.label for m in at.metric] == ["Test MAE", "Test RMSE", "Test R²"]


@needs_model
def test_app_predicts_from_form_and_matches_module():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py")).run(timeout=60)
    at.number_input(key="year").set_value(2014)
    at.number_input(key="present_price").set_value(9.4)
    at.number_input(key="kms").set_value(45_000)
    at.selectbox(key="fuel").set_value("Diesel")
    at.button[0].click().run(timeout=60)
    assert not at.exception

    expected = predict_price(load_model_bundle(), {
        "Year": 2014, "Present_Price": 9.4, "Kms_Driven": 45_000,
        "Fuel_Type": "Diesel", "Transmission": "Manual"}).price
    shown = at.metric[0]
    assert shown.label == "Estimated resale price"
    assert shown.value == f"₹ {expected:.2f} lakh"


@needs_model
def test_app_warns_for_out_of_range_input():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py")).run(timeout=60)
    at.number_input(key="present_price").set_value(90.0)
    at.button[0].click().run(timeout=60)
    assert any("outside the training range" in w.value for w in at.warning)
