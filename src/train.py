"""Train, select, evaluate and save the Multiple Linear Regression model.

Run from the project root:
    python -m src.train

Steps (decisions D7–D10 in docs/decision_log.md):
1. Clean the data and make the fixed 80/20 split.
2. Cross-validate the three pre-declared variants on the TRAINING set and
   pick the lowest mean RMSE. The test set is not touched here.
3. Fit the chosen variant on all training cars and save it.
4. Evaluate once on the test set (all 40 cars, plus the 39 inside the
   training feature ranges).
5. Secondary experiment: the same procedure without Present_Price.

Outputs: models/car_price_model.joblib, reports/model_metrics.json,
reports/coefficients.csv, reports/test_predictions.csv and figures 09–12.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, cross_validate

from src import config
from src.data_preprocessing import clean_data, load_raw_data, split_data
from src.eda import rows_outside_training_range
from src.evaluate import (
    interpret_coefficients,
    plot_actual_vs_predicted,
    plot_cv_comparison,
    plot_residuals,
    regression_metrics,
)
from src.model import VARIANTS, build_pipeline, input_columns

MODEL_PATH = config.MODELS_DIR / "car_price_model.joblib"
CV_SPLITS, CV_REPEATS = 5, 10


def cross_validate_variant(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Mean and spread of MAE, RMSE and R² across repeated K-fold CV."""
    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                       random_state=config.RANDOM_SEED)
    with warnings.catch_warnings():
        # A fold may hold out both CNG cars; they are then scored as Petrol.
        warnings.filterwarnings("ignore", message="Found unknown categories")
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"),
        )
    out = {}
    for key, name in [("test_neg_mean_absolute_error", "MAE"),
                      ("test_neg_root_mean_squared_error", "RMSE"),
                      ("test_r2", "R2")]:
        vals = scores[key] * (-1 if key.startswith("test_neg") else 1)
        out[f"{name}_mean"] = round(float(vals.mean()), 3)
        out[f"{name}_sd"] = round(float(vals.std()), 3)
    return out


def select_variant(cv_results: dict[str, dict]) -> str:
    """Decision D10: lowest mean cross-validated RMSE wins."""
    return min(cv_results, key=lambda v: cv_results[v]["RMSE_mean"])


def out_of_range_index(train: pd.DataFrame, test: pd.DataFrame) -> pd.Index:
    """Test cars outside the training range of the primary model's numeric
    inputs (decision D7). Both experiments use this same set, so their
    within-range metrics cover the same cars."""
    return rows_outside_training_range(
        train, test, ["Year", "Present_Price", "Kms_Driven"]).index


def evaluate_on_test(model, test: pd.DataFrame, cols: list[str],
                     outside: pd.Index) -> dict:
    """Test metrics on all cars and on the cars inside training ranges (D7)."""
    pred = model.predict(test[cols])
    keep = ~test.index.isin(outside)
    return {
        "all": regression_metrics(test[config.TARGET], pred),
        "within_training_range": regression_metrics(test[config.TARGET][keep], pred[keep]),
        "excluded_as_out_of_range": [
            {"car": test.loc[i, "Car_Name"], "actual": float(test.loc[i, config.TARGET]),
             "predicted": round(float(pred[test.index.get_loc(i)]), 2)}
            for i in outside
        ],
    }


def main() -> None:
    clean, _ = clean_data(load_raw_data())
    train, test = split_data(clean)
    y_train = train[config.TARGET]
    cols = input_columns()

    # --- Step 2: model selection on training data only ------------------
    cv_results = {v: cross_validate_variant(build_pipeline(v), train[cols], y_train)
                  for v in VARIANTS}
    chosen = select_variant(cv_results)
    log_target = VARIANTS[chosen]["log_target"]
    print("Cross-validation (training set):")
    print(pd.DataFrame(cv_results).T.to_string())
    print(f"Chosen variant: {chosen}\n")

    # --- Step 3: fit and save -------------------------------------------
    model = build_pipeline(chosen).fit(train[cols], y_train)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "input_columns": cols, "variant": chosen,
                 "reference_year": config.REFERENCE_YEAR,
                 "training_ranges": {c: [float(train[c].min()), float(train[c].max())]
                                     for c in ["Year", "Present_Price", "Kms_Driven"]},
                 "categories": {c: sorted(train[c].unique().tolist())
                                for c in ["Fuel_Type", "Transmission"]}},
                MODEL_PATH)

    # --- Step 4: one evaluation on the test set ---------------------------
    train_metrics = regression_metrics(y_train, model.predict(train[cols]))
    outside = out_of_range_index(train, test)
    test_metrics = evaluate_on_test(model, test, cols, outside)
    test_pred = model.predict(test[cols])

    # --- Step 5: secondary experiment without Present_Price --------------
    # log_log needs Present_Price, so without it the chosen target scale is used.
    variant_np = "log_target" if log_target else "raw"
    cols_np = input_columns(include_present_price=False)
    model_np = build_pipeline(variant_np, include_present_price=False)
    cv_np = cross_validate_variant(clone(model_np), train[cols_np], y_train)
    model_np.fit(train[cols_np], y_train)
    no_present_price = {
        "variant": variant_np,
        "features": cols_np,
        "cv_train": cv_np,
        "train": regression_metrics(y_train, model_np.predict(train[cols_np])),
        "test": evaluate_on_test(model_np, test, cols_np, outside),
    }

    # --- Outputs -----------------------------------------------------------
    coefs = interpret_coefficients(model, log_target)
    coefs.to_csv(config.REPORTS_DIR / "coefficients.csv", index=False)

    predictions = test[["Car_Name", "Year", "Present_Price", "Kms_Driven",
                        "Fuel_Type", "Transmission", config.TARGET]].copy()
    predictions["Predicted"] = np.round(test_pred, 2)
    predictions["Abs_Error"] = np.round(np.abs(predictions[config.TARGET] - test_pred), 2)
    predictions = predictions.sort_values("Abs_Error")
    predictions.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)

    figures = [
        plot_cv_comparison(
            {**{f"{v} (with Present_Price)": r for v, r in cv_results.items()},
             f"{variant_np} (without Present_Price)": cv_np},
            "09_cv_comparison.png"),
        plot_actual_vs_predicted(test[config.TARGET].reset_index(drop=True), test_pred,
                                 test["Car_Name"].reset_index(drop=True),
                                 "10_actual_vs_predicted.png"),
        *plot_residuals(test[config.TARGET], test_pred,
                        "11_residual_distribution.png", "12_residuals_vs_predicted.png"),
    ]

    metrics = {
        "model": "sklearn.linear_model.LinearRegression",
        "variant": chosen,
        "features": cols,
        "split": {"train": len(train), "test": len(test),
                  "test_size": config.TEST_SIZE, "random_seed": config.RANDOM_SEED},
        "cv_scheme": f"RepeatedKFold({CV_SPLITS} splits x {CV_REPEATS} repeats) on training set",
        "cv_candidates": cv_results,
        "train": train_metrics,
        "test": test_metrics,
        "secondary_experiment_without_present_price": no_present_price,
        "units": "MAE and RMSE in ₹ lakhs",
        "figures": figures,
    }
    (config.REPORTS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in
                      ["train", "test", "secondary_experiment_without_present_price"]},
                     indent=2, ensure_ascii=False))
    print("\nCoefficients:\n" + coefs.to_string(index=False))
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
