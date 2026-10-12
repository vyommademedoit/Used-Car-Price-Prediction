"""Load the saved model and make validated predictions.

Used by the Streamlit app (app.py); can also be used from Python:

    from src.predict import load_model_bundle, predict_price
    bundle = load_model_bundle()
    result = predict_price(bundle, {"Year": 2015, "Present_Price": 8.5,
                                    "Kms_Driven": 40000, "Fuel_Type": "Petrol",
                                    "Transmission": "Manual"})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.train import MODEL_PATH

# Hard limits: values outside these are rejected as impossible or nonsensical.
# MAX_YEAR is the reference year: newer cars would get a negative age, and the
# data (listings up to about 2020) says nothing about them.
MIN_YEAR, MAX_YEAR = config.MIN_YEAR, config.REFERENCE_YEAR
MAX_PRESENT_PRICE = 500.0   # ₹ lakh; far above any car in the data
MAX_KMS = 1_000_000


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file has not been created yet."""


@dataclass
class Prediction:
    price: float | None                          # ₹ lakh, None if inputs invalid
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    """Load the dict saved by `python -m src.train`."""
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.train` "
            "from the project root first (see README)."
        )
    bundle = joblib.load(path)
    missing = {"model", "input_columns", "training_ranges", "categories"} - set(bundle)
    if missing:
        raise ValueError(f"Model file is missing {sorted(missing)}; retrain it.")
    return bundle


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_inputs(bundle: dict, inputs: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).

    Errors block the prediction. Warnings mean the inputs are valid but lie
    outside the data the model was trained on, so the estimate is less
    trustworthy.
    """
    errors, warnings = [], []
    for col in bundle["input_columns"]:
        if inputs.get(col) is None or inputs.get(col) == "":
            errors.append(f"{col} is required.")
    if errors:
        return errors, warnings

    year, price, kms = inputs["Year"], inputs["Present_Price"], inputs["Kms_Driven"]
    if not _is_number(year) or float(year) != int(float(year)):
        errors.append("Year must be a whole number.")
    elif not MIN_YEAR <= int(float(year)) <= MAX_YEAR:
        errors.append(f"Year must be between {MIN_YEAR} and {MAX_YEAR} "
                      "(the data only covers cars listed up to about 2020).")
    if not _is_number(price) or not 0 < float(price) <= MAX_PRESENT_PRICE:
        errors.append(f"Present price must be a number above 0 and at most "
                      f"{MAX_PRESENT_PRICE:g} lakh.")
    if not _is_number(kms) or not 0 <= float(kms) <= MAX_KMS:
        errors.append(f"Kilometres driven must be between 0 and {MAX_KMS:,}.")
    for col, allowed in bundle["categories"].items():
        if inputs[col] not in allowed:
            errors.append(f"{col} must be one of {allowed}.")
    if errors:
        return errors, warnings

    labels = {"Year": "Year", "Present_Price": "Present price (₹ lakh)",
              "Kms_Driven": "Kilometres driven"}
    for col, (lo, hi) in bundle["training_ranges"].items():
        if not lo <= float(inputs[col]) <= hi:
            warnings.append(f"{labels[col]} {float(inputs[col]):g} is outside the "
                            f"training range {lo:g}–{hi:g}; the model is extrapolating.")
    if inputs["Fuel_Type"] == "CNG":
        warnings.append("Only 2 CNG cars were in the training data, so the CNG "
                        "adjustment is unreliable.")
    return errors, warnings


def predict_price(bundle: dict, inputs: dict) -> Prediction:
    """Validate inputs and return the estimated selling price in ₹ lakh."""
    errors, warnings = validate_inputs(bundle, inputs)
    if errors:
        return Prediction(price=None, errors=errors, warnings=warnings)

    row = pd.DataFrame([{
        "Year": int(float(inputs["Year"])),
        "Present_Price": float(inputs["Present_Price"]),
        "Kms_Driven": float(inputs["Kms_Driven"]),
        "Fuel_Type": inputs["Fuel_Type"],
        "Transmission": inputs["Transmission"],
    }])[bundle["input_columns"]]
    price = float(bundle["model"].predict(row)[0])

    if price > float(inputs["Present_Price"]):
        warnings.append("The estimate is above the present (new-car) price, "
                        "which is implausible for a used car. Check the inputs.")
    return Prediction(price=price, warnings=warnings)
