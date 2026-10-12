"""Metrics, diagnostic plots and coefficient interpretation.

All metrics are computed in ₹ lakhs, even for models trained on log price,
so every variant is compared in the same units.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.eda import BLUE, INK, INK_MUTED, _save, _scatter, _style
from src.model import REFERENCE_LEVELS, coefficient_table


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE (both ₹ lakhs) and R² for one set of predictions."""
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
        "n": int(len(y_true)),
    }


def interpret_coefficients(model, log_target: bool) -> pd.DataFrame:
    """Coefficient table with a plain-language effect for each term.

    On a log target, a coefficient b means "multiply the price by e^b", so
    the effect is shown as a percentage change: (e^b − 1) × 100.
    Kilometres are shown per 10,000 km because per-km effects are tiny.
    """
    table = coefficient_table(model)
    effects = []
    for term, b in zip(table["term"], table["coefficient"]):
        if term == "(intercept)":
            effects.append("Baseline value; not meaningful alone (no car has all inputs = 0)")
            continue
        if term == "log_Present_Price":
            effects.append(f"+1% present price → {b:+.2f}% selling price")
            continue
        scale = 10_000 if term == "Kms_Driven" else 1
        if term == "Car_Age":
            what = "per extra year of age"
        elif term == "Kms_Driven":
            what = "per 10,000 km"
        elif term == "Present_Price":
            what = "per ₹1 lakh of present price"
        else:
            what = _vs_reference(term)
        if log_target:
            effects.append(f"{(np.exp(b * scale) - 1) * 100:+.1f}% selling price {what}")
        else:
            effects.append(f"{b * scale:+.2f} lakh {what}")
    table["effect"] = effects
    return table


def _vs_reference(term: str) -> str:
    col, level = term.rsplit("_", 1)
    return f"for {level} vs {REFERENCE_LEVELS[col]} (other inputs equal)"


def plot_actual_vs_predicted(y_true, y_pred, labels: pd.Series, name: str) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _scatter(ax, y_true, y_pred)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, color=INK_MUTED, ls="--", lw=1, label="Perfect prediction")
    for i in np.argsort(np.abs(y_true - y_pred))[-3:]:
        ax.annotate(labels.iloc[i], (y_true.iloc[i], y_pred[i]), fontsize=8,
                    xytext=(5, -10), textcoords="offset points", color=INK)
    ax.set(xlim=lim, ylim=lim, xlabel="Actual selling price (₹ lakh)",
           ylabel="Predicted selling price (₹ lakh)",
           title="Test set: actual vs predicted")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, name)


def plot_residuals(y_true, y_pred, name_hist: str, name_scatter: str) -> list[str]:
    _style()
    resid = y_true.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(resid, bins=20, color=BLUE, edgecolor="#fcfcfb")
    ax.axvline(0, color=INK, lw=1)
    ax.set(title="Test residuals (actual − predicted)",
           xlabel="Residual (₹ lakh)", ylabel="Number of cars")
    first = _save(fig, name_hist)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _scatter(ax, y_pred, resid)
    ax.axhline(0, color=INK, lw=1)
    ax.set(title="Test residuals vs predicted price",
           xlabel="Predicted selling price (₹ lakh)", ylabel="Residual (₹ lakh)")
    return [first, _save(fig, name_scatter)]


def plot_cv_comparison(cv_results: dict[str, dict], name: str) -> str:
    """Mean CV RMSE (± 1 sd across folds) for each candidate."""
    _style()
    names = list(cv_results)
    means = [cv_results[n]["RMSE_mean"] for n in names]
    sds = [cv_results[n]["RMSE_sd"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.6 * len(names)))
    ax.barh(names, means, xerr=sds, color=BLUE, height=0.5,
            error_kw={"ecolor": INK_MUTED, "lw": 1, "capsize": 3})
    for y, m in enumerate(means):
        ax.text(m / 2, y, f"{m:.2f}", va="center", ha="center", color="white",
                fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set(xlabel="Cross-validated RMSE (₹ lakh, lower is better)",
           title="Training-set cross-validation (5-fold × 10 repeats)")
    return _save(fig, name)

