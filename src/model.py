"""Build the Multiple Linear Regression pipeline.

The pipeline takes the raw columns a user can supply (Year, Present_Price,
Kms_Driven, Fuel_Type, Transmission) and does every transformation itself.
Training, evaluation and the Streamlit app therefore all use exactly the
same preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src import config

# Model input schema (decision D8). Seller_Type and Owner are excluded.
NUMERIC_INPUTS = ["Year", "Present_Price", "Kms_Driven"]
CATEGORICAL_INPUTS = ["Fuel_Type", "Transmission"]
# Reference (dropped) level of each category: the most common one, so every
# coefficient reads as "compared with a petrol manual car" (decision D9).
REFERENCE_LEVELS = {"Fuel_Type": "Petrol", "Transmission": "Manual"}

# Pre-declared candidates compared by cross-validation (decision D10).
#   log_target: model log(Selling_Price) and convert predictions back.
#   log_present_price: use log(Present_Price) as the input instead.
VARIANTS = {
    "raw": {"log_target": False, "log_present_price": False},
    "log_target": {"log_target": True, "log_present_price": False},
    "log_log": {"log_target": True, "log_present_price": True},
}


def input_columns(include_present_price: bool = True) -> list[str]:
    cols = [*NUMERIC_INPUTS, *CATEGORICAL_INPUTS]
    return cols if include_present_price else [c for c in cols if c != "Present_Price"]


# Named functions (not lambdas) so the fitted pipeline can be saved with joblib.
def year_to_age(year):
    return config.REFERENCE_YEAR - year


def age_feature_name(transformer, input_features):
    return ["Car_Age"]


def log_feature_name(transformer, input_features):
    return [f"log_{name}" for name in input_features]


def build_pipeline(
    variant: str = "log_log", include_present_price: bool = True
) -> TransformedTargetRegressor | Pipeline:
    """Return an unfitted preprocessing + LinearRegression pipeline.

    No scaling is applied: ordinary least squares does not need it, and
    unscaled inputs keep each coefficient in real units (years, km, lakhs).
    """
    opts = VARIANTS[variant]
    transformers = [
        ("age", FunctionTransformer(year_to_age, feature_names_out=age_feature_name),
         ["Year"]),
        ("kms", "passthrough", ["Kms_Driven"]),
    ]
    if include_present_price:
        price = (FunctionTransformer(np.log, feature_names_out=log_feature_name)
                 if opts["log_present_price"] else "passthrough")
        transformers.append(("price", price, ["Present_Price"]))
    transformers.append((
        "cat",
        OneHotEncoder(
            drop=[REFERENCE_LEVELS[c] for c in CATEGORICAL_INPUTS],
            handle_unknown="ignore",  # unseen level -> treated as the reference
        ),
        CATEGORICAL_INPUTS,
    ))
    pipe = Pipeline([
        ("preprocess", ColumnTransformer(transformers, verbose_feature_names_out=False)),
        ("regression", LinearRegression()),
    ])
    if opts["log_target"]:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def fitted_linear_part(model) -> Pipeline:
    """The inner fitted Pipeline, whether or not the target was log-transformed."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def coefficient_table(model) -> pd.DataFrame:
    """Intercept and coefficients of a fitted pipeline, with feature names."""
    pipe = fitted_linear_part(model)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    reg = pipe.named_steps["regression"]
    return pd.DataFrame({
        "term": ["(intercept)", *names],
        "coefficient": [reg.intercept_, *reg.coef_],
    })
