"""Exploratory data analysis on the TRAINING split only.

Run from the project root (after `python -m src.data_preprocessing`):
    python -m src.eda

Writes figures to reports/figures/ and the numbers quoted in
docs/eda_findings.md to reports/eda_summary.json.

Why training data only (decision D6): what we see in EDA shapes later
modelling choices (feature set, target transform). Looking at the test
rows' prices here would let test information leak into those choices.
The only test-set check made is on its *feature* ranges, to flag
predictions that will be extrapolations; no test prices are used.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data_preprocessing import add_car_age, clean_data, load_raw_data, split_data

PRICE_LABEL = "Selling price (₹ lakh)"
NUMERIC_FEATURES = ["Present_Price", "Kms_Driven", "Car_Age", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Seller_Type", "Transmission"]

# Colours: validated reference palette (first three categorical slots are
# colour-blind safe as a set); neutral inks for text, grid and baselines.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = sns.blend_palette(["#2a78d6", "#f0efec", "#e34948"], as_cmap=True)


# --- Numerical helpers (unit-tested) -------------------------------------

def design_matrix(df: pd.DataFrame, log_prices: bool = False) -> pd.DataFrame:
    """Numeric features plus one-hot categoricals (first level dropped).

    Used only for EDA diagnostics (VIF, Cook's distance), not for the model.
    With log_prices=True, Present_Price is log-transformed.
    """
    X = df[NUMERIC_FEATURES].astype(float).copy()
    if log_prices:
        X["Present_Price"] = np.log(X["Present_Price"])
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True, dtype=float)
    return pd.concat([X, dummies], axis=1)


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R²_j), where R²_j regresses column j on the others.

    Values above about 5 suggest a coefficient will be unstable because the
    feature is largely explained by the other features.
    """
    vifs = {}
    for col in X.columns:
        y = X[col].to_numpy()
        others = np.column_stack([np.ones(len(X)), X.drop(columns=col).to_numpy()])
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ coef
        r2 = 1 - resid.var() / y.var()
        vifs[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(vifs)


def cooks_distance(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Cook's distance of each row for an ordinary least-squares fit of y on X.

    It measures how much all fitted values move when that one row is left
    out. A common rule of thumb flags rows above 4 / n for a closer look.
    """
    A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    n, p = A.shape
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(dtype=float), rcond=None)
    resid = y.to_numpy(dtype=float) - A @ coef
    leverage = np.einsum("ij,ij->i", A @ np.linalg.pinv(A.T @ A), A)
    mse = resid @ resid / (n - p)
    return resid**2 / (p * mse) * leverage / (1 - leverage) ** 2


def rows_outside_training_range(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Test rows with any feature value outside the training min-max range.

    Uses test *features* only. A linear model extrapolates for these rows.
    """
    outside = pd.Series(False, index=test.index)
    for col in columns:
        outside |= ~test[col].between(train[col].min(), train[col].max())
    return test[outside]


# --- Plotting -----------------------------------------------------------

def _style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#c3c2b7", "grid.color": GRID, "grid.linewidth": 0.6,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "axes.titleweight": "bold", "axes.titlesize": 11,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })


def _save(fig: plt.Figure, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return name


def _scatter(ax, x, y, color=BLUE, **kw):
    ax.scatter(x, y, s=28, color=color, alpha=0.75, edgecolor="#fcfcfb",
               linewidth=0.8, **kw)


def plot_price_distribution(train: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(train["Selling_Price"], bins=25, color=BLUE, ax=axes[0])
    axes[0].axvline(train["Selling_Price"].median(), color=INK, ls="--", lw=1)
    axes[0].set(title="Selling price: long right tail", xlabel=PRICE_LABEL,
                ylabel="Number of cars")
    sns.histplot(np.log(train["Selling_Price"]), bins=25, color=BLUE, ax=axes[1])
    axes[1].set(title="log(Selling price): roughly symmetric",
                xlabel="log(selling price in ₹ lakh)", ylabel="Number of cars")
    return _save(fig, "01_price_distribution.png")


def plot_price_vs_age(train: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _scatter(axes[0], train["Car_Age"], train["Selling_Price"])
    med = train.groupby("Car_Age")["Selling_Price"].median()
    axes[0].plot(med.index, med.values, color=INK, lw=2, label="Median per age")
    axes[0].legend(frameon=False)
    axes[0].set(title="Selling price vs car age", xlabel="Car age (years, 2020 − Year)",
                ylabel=PRICE_LABEL)

    retained = train["Selling_Price"] / train["Present_Price"]
    _scatter(axes[1], train["Car_Age"], retained)
    med = retained.groupby(train["Car_Age"]).median()
    axes[1].plot(med.index, med.values, color=INK, lw=2)
    axes[1].set(title="Share of present price retained vs age",
                xlabel="Car age (years)", ylabel="Selling ÷ Present price", ylim=(0, 1.05))
    return _save(fig, "02_price_vs_age.png")


def plot_price_vs_kms(train: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _scatter(ax, train["Kms_Driven"] / 1000, train["Selling_Price"])
    ax.set(title="Selling price vs kilometres driven",
           xlabel="Kilometres driven (thousands)", ylabel=PRICE_LABEL)
    return _save(fig, "03_price_vs_kms.png")


def _box_with_points(train: pd.DataFrame, col: str, order: list[str], ax) -> None:
    sns.boxplot(data=train, x=col, y="Selling_Price", order=order, ax=ax,
                color="#cde2fb", width=0.5, fliersize=0, linewidth=1)
    sns.stripplot(data=train, x=col, y="Selling_Price", order=order, ax=ax,
                  color=BLUE, size=4, alpha=0.7, jitter=0.2)
    counts = train[col].value_counts()
    ax.set_xticks(range(len(order)), [f"{c}\n(n={counts.get(c, 0)})" for c in order])
    ax.set(xlabel="", ylabel=PRICE_LABEL)


def plot_price_by_category(train: pd.DataFrame, col: str, order: list[str],
                           name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _box_with_points(train, col, order, axes[0])
    axes[0].set_title(f"Selling price by {col.replace('_', ' ').lower()}")
    sns.boxplot(data=train, x=col, y="Present_Price", order=order, ax=axes[1],
                color="#ffd9c6", width=0.5, fliersize=0, linewidth=1)
    sns.stripplot(data=train, x=col, y="Present_Price", order=order, ax=axes[1],
                  color=ORANGE, size=4, alpha=0.7, jitter=0.2)
    axes[1].set(title="…and present (new-car) price, same groups", xlabel="",
                ylabel="Present price (₹ lakh)")
    return _save(fig, name)


def plot_correlations(train: pd.DataFrame) -> str:
    cols = ["Selling_Price", *NUMERIC_FEATURES]
    corr = train[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1,
                square=True, linewidths=2, linecolor="#fcfcfb", ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation between numeric columns (training set)")
    return _save(fig, "06_correlation_heatmap.png")


def plot_present_vs_selling(train: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"Manual": BLUE, "Automatic": ORANGE}
    for trans, group in train.groupby("Transmission"):
        _scatter(ax, group["Present_Price"], group["Selling_Price"],
                 color=colors[trans], label=trans)
    lim = [1, train["Present_Price"].max() * 1.1]
    ax.plot(lim, lim, color=INK_MUTED, ls="--", lw=1, label="Selling = Present")
    ax.set(xscale="log", yscale="log", xlabel="Present price (₹ lakh, log scale)",
           ylabel="Selling price (₹ lakh, log scale)",
           title="Selling vs present price (log–log)")
    ticks = [0.5, 1, 2, 5, 10, 20, 40]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.legend(frameon=False)
    return _save(fig, "07_present_vs_selling.png")


def plot_influence(train: pd.DataFrame, cooks_raw: np.ndarray,
                   cooks_log: np.ndarray, threshold: float) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, d, title in [
        (axes[0], cooks_raw, "Linear fit on raw prices"),
        (axes[1], cooks_log, "Linear fit on log prices"),
    ]:
        x = np.arange(len(d))
        ax.vlines(x, 0, d, color=BLUE, lw=1.2)
        ax.axhline(threshold, color=INK_MUTED, ls="--", lw=1,
                   label=f"4/n = {threshold:.3f}")
        for i in np.argsort(d)[-3:]:
            row = train.iloc[i]
            ax.annotate(f"{row.Car_Name} {row.Year}", (i, d[i]), fontsize=8,
                        xytext=(4, 2), textcoords="offset points", color=INK)
        ax.set(title=f"Cook's distance — {title}", xlabel="Training row",
               ylabel="Cook's distance")
        ax.legend(frameon=False, loc="upper left")
    return _save(fig, "08_influence_cooks_distance.png")


# --- Orchestration ------------------------------------------------------

def run_eda() -> dict:
    """Produce every figure and return the summary numbers."""
    _style()
    clean, _ = clean_data(load_raw_data())
    train, test = split_data(add_car_age(clean))
    train = train.reset_index(drop=True)

    X_raw = design_matrix(train)
    X_log = design_matrix(train, log_prices=True)
    cooks_raw = cooks_distance(X_raw, train["Selling_Price"])
    cooks_log = cooks_distance(X_log, np.log(train["Selling_Price"]))
    threshold = 4 / len(train)

    def top_influential(d: np.ndarray) -> list[dict]:
        idx = np.argsort(d)[::-1][:5]
        cols = ["Car_Name", "Year", "Selling_Price", "Present_Price", "Kms_Driven"]
        return [
            {**{c: (v.item() if hasattr(v, "item") else v) for c, v in
                train.loc[i, cols].items()}, "cooks_distance": round(float(d[i]), 3)}
            for i in idx
        ]

    figures = [
        plot_price_distribution(train),
        plot_price_vs_age(train),
        plot_price_vs_kms(train),
        plot_price_by_category(train, "Fuel_Type", ["Petrol", "Diesel", "CNG"],
                               "04_price_by_fuel.png"),
        plot_price_by_category(train, "Transmission", ["Manual", "Automatic"],
                               "05_price_by_transmission.png"),
        plot_correlations(train),
        plot_present_vs_selling(train),
        plot_influence(train, cooks_raw, cooks_log, threshold),
    ]

    retained = train["Selling_Price"] / train["Present_Price"]
    numeric = train[["Selling_Price", *NUMERIC_FEATURES]]
    outside = rows_outside_training_range(
        train, test, ["Present_Price", "Kms_Driven", "Car_Age"]
    )

    def group_stats(col: str) -> dict:
        g = train.groupby(col)
        return {
            str(k): {
                "n": int(len(v)),
                "median_selling_price": round(float(v["Selling_Price"].median()), 2),
                "median_present_price": round(float(v["Present_Price"].median()), 2),
                "median_share_retained": round(float(
                    (v["Selling_Price"] / v["Present_Price"]).median()), 2),
            }
            for k, v in g
        }

    summary = {
        "rows_train": len(train),
        "rows_test": len(test),
        "selling_price": {
            **{k: round(float(v), 2) for k, v in
               train["Selling_Price"].describe().items() if k != "count"},
            "skewness": round(float(train["Selling_Price"].skew()), 2),
            "skewness_of_log": round(float(np.log(train["Selling_Price"]).skew()), 2),
        },
        "pearson_with_target": numeric.corr()["Selling_Price"].drop("Selling_Price")
            .round(2).to_dict(),
        "spearman_with_target": numeric.corr("spearman")["Selling_Price"]
            .drop("Selling_Price").round(2).to_dict(),
        "corr_car_age_kms": round(float(train["Car_Age"].corr(train["Kms_Driven"])), 2),
        "corr_car_age_present_price": round(float(
            train["Car_Age"].corr(train["Present_Price"])), 2),
        "share_retained": {
            "median": round(float(retained.median()), 2),
            "corr_with_car_age": round(float(retained.corr(train["Car_Age"])), 2),
            "median_by_age": retained.groupby(train["Car_Age"]).median()
                .round(2).to_dict(),
        },
        "by_fuel_type": group_stats("Fuel_Type"),
        "by_transmission": group_stats("Transmission"),
        "by_seller_type": group_stats("Seller_Type"),
        "by_owner": group_stats("Owner"),
        "vif_raw_features": variance_inflation_factors(X_raw).round(2).to_dict(),
        "cooks_threshold_4_over_n": round(threshold, 4),
        "cooks_rows_above_threshold": {
            "raw_price_fit": int((cooks_raw > threshold).sum()),
            "log_price_fit": int((cooks_log > threshold).sum()),
        },
        "most_influential_raw_price_fit": top_influential(cooks_raw),
        "most_influential_log_price_fit": top_influential(cooks_log),
        "training_feature_ranges": {
            c: [float(train[c].min()), float(train[c].max())]
            for c in ["Present_Price", "Kms_Driven", "Car_Age"]
        },
        "test_rows_outside_training_feature_range": [
            {c: (v.item() if hasattr(v, "item") else v) for c, v in
             row[["Car_Name", "Year", "Present_Price", "Kms_Driven", "Car_Age"]].items()}
            for _, row in outside.iterrows()
        ],
        "minority_levels_in_test": {
            "Fuel_Type=CNG": int((test["Fuel_Type"] == "CNG").sum()),
            "Seller_Type=Individual": int((test["Seller_Type"] == "Individual").sum()),
            "Owner>0": int((test["Owner"] > 0).sum()),
        },
        "figures": figures,
    }
    return summary


def main() -> None:
    # Old Windows consoles (cp1252) cannot print characters such as "→" or "₹".
    sys.stdout.reconfigure(errors="replace")
    summary = run_eda()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(summary['figures'])} figures to {config.FIGURES_DIR}")
    print(f"Saved summary to {path}")


if __name__ == "__main__":
    main()
