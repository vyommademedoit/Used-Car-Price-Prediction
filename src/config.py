"""Project-wide settings in one place.

Paths are built relative to the project root so the code runs on any
machine without editing absolute paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cars_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Experiment settings (kept explicit for reproducibility).
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Car age = REFERENCE_YEAR - Year. In a linear model, changing this value
# only shifts the intercept; coefficients and predictions are unchanged.
# 2020 is used because the newest car in the data is from 2018 and the
# dataset was published around then.
REFERENCE_YEAR = 2020

TARGET = "Selling_Price"

# Columns the raw CSV must contain, with the kind of values expected.
REQUIRED_COLUMNS = {
    "Car_Name": "text",
    "Year": "integer",
    "Selling_Price": "number",   # ₹ lakhs
    "Present_Price": "number",   # ₹ lakhs (current ex-showroom price)
    "Kms_Driven": "integer",
    "Fuel_Type": "category",
    "Seller_Type": "category",
    "Transmission": "category",
    "Owner": "integer",          # number of previous owners
}

ALLOWED_CATEGORIES = {
    "Fuel_Type": {"Petrol", "Diesel", "CNG"},
    "Seller_Type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}

MIN_YEAR = 1990

# Name prefixes that identify two-wheelers mixed into this "car" dataset.
# Car rows use bare model names ("city", "fortuner", "800"), whereas every
# two-wheeler row starts with a manufacturer or scooter name.
TWO_WHEELER_PREFIXES = (
    "activa", "bajaj", "hero", "honda", "hyosung", "ktm", "mahindra mojo",
    "royal enfield", "suzuki access", "tvs", "um ", "yamaha",
)
