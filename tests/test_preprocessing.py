"""Tests for src/data_preprocessing.py.

Most tests use a small hand-made DataFrame so they run without the real
dataset. The final tests use the real CSV and are skipped if it is absent.
"""

import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    SchemaError,
    clean_data,
    is_two_wheeler,
    load_raw_data,
    validate_schema,
)


def make_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame where each row starts from a valid car."""
    base = {
        "Car_Name": "city", "Year": 2015, "Selling_Price": 5.0,
        "Present_Price": 8.0, "Kms_Driven": 30000, "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_missing_column_raises_schema_error():
    df = make_rows([{}]).drop(columns=["Kms_Driven"])
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_non_numeric_price_raises_schema_error():
    df = make_rows([{"Selling_Price": "five"}])
    with pytest.raises(SchemaError):
        validate_schema(df)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="car_data.csv"):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_invalid_values_are_removed():
    df = make_rows([
        {},                              # valid
        {"Selling_Price": -1.0},         # impossible price
        {"Kms_Driven": -5},              # impossible km
        {"Year": 2050},                  # future year
        {"Fuel_Type": "Electric"},       # unknown category
        {"Present_Price": None},         # missing value
    ])
    clean, report = clean_data(df)
    assert len(clean) == 1
    assert report["rows_removed_invalid"] == 5


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Royal Enfield Classic 350", True),
        ("Bajaj  Pulsar NS 200", True),   # double space in raw data
        ("Yamaha Fazer ", True),          # trailing space in raw data
        ("Activa 3g", True),
        ("city", False),
        ("800", False),                   # Maruti 800, a car
        ("land cruiser", False),
    ],
)
def test_two_wheeler_detection(name, expected):
    assert is_two_wheeler(name) is expected


def test_two_wheelers_removed_by_default_but_can_be_kept():
    df = make_rows([{}, {"Car_Name": "Hero Splender Plus", "Selling_Price": 0.3}])
    assert len(clean_data(df)[0]) == 1
    assert len(clean_data(df, keep_two_wheelers=True)[0]) == 2


def test_exact_duplicates_removed():
    df = make_rows([{}, {}, {"Kms_Driven": 40000}])
    clean, report = clean_data(df)
    assert len(clean) == 2
    assert report["rows_removed_duplicates"] == 1


def test_clean_data_does_not_modify_input():
    df = make_rows([{"Car_Name": "  City "}])
    clean_data(df)
    assert df.loc[0, "Car_Name"] == "  City "


# --- Tests against the real dataset -------------------------------------

real_data = pytest.mark.skipif(
    not config.RAW_DATA_PATH.exists(), reason="raw dataset not downloaded"
)


@real_data
def test_real_dataset_schema_and_size():
    df = load_raw_data()
    assert list(df.columns) == list(config.REQUIRED_COLUMNS)
    assert len(df) == 301


@real_data
def test_real_dataset_cleaning_counts():
    clean, report = clean_data(load_raw_data())
    assert report["two_wheelers_found"] == 101
    assert report["rows_removed_duplicates"] == 2
    assert report["rows_clean"] == 198
    assert not clean["Car_Name"].map(is_two_wheeler).any()
    assert clean[list(config.REQUIRED_COLUMNS)].notna().all().all()


@real_data
def test_two_wheeler_list_matches_independent_naming_check():
    """In this file, car names are all lower-case and two-wheeler names are
    not. Use that as an independent cross-check of the prefix list."""
    df = load_raw_data()
    by_prefix = df["Car_Name"].map(is_two_wheeler)
    by_casing = df["Car_Name"] != df["Car_Name"].str.lower()
    assert (by_prefix == by_casing).all()
