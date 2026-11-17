"""Load, validate and clean the CarDekho `car data.csv` file.

Run from the project root:
    python -m src.data_preprocessing

This writes the cleaned data to data/processed/cars_clean.csv and a summary
of every cleaning step to reports/data_cleaning_report.json.

Only deterministic, row-level cleaning happens here (no statistics are
learned from the data), so running it before the train/test split does not
leak test information. Encoding and anything data-dependent is done later
inside the scikit-learn Pipeline, fitted on training data only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src import config


class SchemaError(ValueError):
    """Raised when the CSV does not have the columns the project expects."""


def load_raw_data(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that every required column is present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'car data.csv' from the "
            "Kaggle dataset 'Vehicle dataset from CarDekho' (nehalbirla) and "
            "save it as data/raw/car_data.csv."
        )
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError if required columns are missing or not numeric."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    numeric_cols = [
        c for c, kind in config.REQUIRED_COLUMNS.items()
        if kind in {"integer", "number"}
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"Column '{col}' should be numeric, got {df[col].dtype}")


def normalise_name(name: str) -> str:
    """Lower-case a vehicle name and collapse repeated/trailing spaces."""
    return " ".join(str(name).lower().split())


def is_two_wheeler(name: str) -> bool:
    """True if the vehicle name identifies a motorcycle or scooter."""
    return normalise_name(name).startswith(config.TWO_WHEELER_PREFIXES)


def find_invalid_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows with impossible or unexpected values."""
    invalid = df[list(config.REQUIRED_COLUMNS)].isna().any(axis=1)
    invalid |= df["Selling_Price"] <= 0
    invalid |= df["Present_Price"] <= 0
    invalid |= df["Kms_Driven"] < 0
    invalid |= df["Owner"] < 0
    invalid |= ~df["Year"].between(config.MIN_YEAR, config.REFERENCE_YEAR)
    for col, allowed in config.ALLOWED_CATEGORIES.items():
        invalid |= ~df[col].isin(allowed)
    return invalid


def clean_data(
    df: pd.DataFrame, keep_two_wheelers: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning steps and return (clean_df, report).

    Steps, in order:
    1. Tidy whitespace in text columns.
    2. Remove rows with missing or impossible values.
    3. Remove two-wheelers (the project is about cars), unless asked not to.
    4. Remove exact duplicate rows, so the same listing cannot end up in
       both the training and test sets.
    """
    df = df.copy()
    report: dict = {"rows_raw": len(df)}

    text_cols = ["Car_Name", *config.ALLOWED_CATEGORIES]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
    df["Car_Name"] = df["Car_Name"].map(normalise_name)

    invalid = find_invalid_rows(df)
    report["rows_removed_invalid"] = int(invalid.sum())
    df = df[~invalid]

    two_wheeler = df["Car_Name"].map(is_two_wheeler)
    report["two_wheelers_found"] = int(two_wheeler.sum())
    if not keep_two_wheelers:
        df = df[~two_wheeler]
    report["rows_removed_two_wheelers"] = (
        0 if keep_two_wheelers else int(two_wheeler.sum())
    )

    duplicated = df.duplicated()
    report["rows_removed_duplicates"] = int(duplicated.sum())
    df = df[~duplicated].reset_index(drop=True)

    report["rows_clean"] = len(df)
    report["rows_selling_above_present_price"] = int(
        (df["Selling_Price"] > df["Present_Price"]).sum()
    )
    report["category_counts"] = {
        col: {str(k): int(v) for k, v in df[col].value_counts().items()}
        for col in [*config.ALLOWED_CATEGORIES, "Owner"]
    }
    return df, report


def add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Car_Age = REFERENCE_YEAR - Year (see decision D5)."""
    df = df.copy()
    df["Car_Age"] = config.REFERENCE_YEAR - df["Year"]
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned data into (train, test) with the project's fixed seed.

    EDA (Stage 2) and model training (Stage 3) both call this, so they see
    exactly the same training rows and the test set stays untouched.
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED
    )
    return train, test


def main() -> None:
    # Old Windows consoles (cp1252) cannot print characters such as "→" or "₹".
    sys.stdout.reconfigure(errors="replace")
    """Run the cleaning pipeline end to end and save its outputs."""
    raw = load_raw_data()
    clean, report = clean_data(raw)

    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.PROCESSED_DATA_PATH, index=False)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "data_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved cleaned data to {config.PROCESSED_DATA_PATH}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
