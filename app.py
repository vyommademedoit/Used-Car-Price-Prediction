"""Streamlit app: estimate a used car's resale price.

Run from the project root (after `python -m src.train`):
    streamlit run app.py

The app only loads the saved pipeline; it never retrains.
"""

import json

import streamlit as st

from src import config
from src.predict import ModelNotFoundError, load_model_bundle, predict_price

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")


@st.cache_resource
def get_bundle() -> dict:
    """Load the model once per server process, not on every interaction."""
    return load_model_bundle()


def load_metrics() -> dict | None:
    path = config.REPORTS_DIR / "model_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --- A. Introduction ------------------------------------------------------
st.title("Indian Used Car Resale Price Estimator")
st.write(
    "Estimates the resale price of a used car from its age, kilometres driven, "
    "fuel type, transmission and current showroom price. It was built from "
    "198 CarDekho car listings from around 2018–2020 (158 used for training, "
    "40 held back for testing)."
)
with st.expander("How does Multiple Linear Regression work here?"):
    st.markdown(
        "The model predicts the **logarithm** of the selling price as a weighted "
        "sum of the inputs:\n\n"
        "`log(price) = b0 + b1·age + b2·km + b3·log(present price) + b4·diesel + …`\n\n"
        "Each weight *b* was learned from training cars by least squares, which "
        "picks the weights that minimise the squared prediction errors. Working "
        "on the log scale means each input changes the price by a *percentage*. "
        "For example, each extra year of age lowers the estimate by about 10%."
    )

try:
    bundle = get_bundle()
except (ModelNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

ranges = bundle["training_ranges"]

# --- B. Prediction form ---------------------------------------------------
st.header("Car details")
with st.form("car_form"):
    col1, col2 = st.columns(2)
    year = col1.number_input(
        "Manufacturing year", min_value=1990, max_value=config.REFERENCE_YEAR,
        value=2015, step=1, key="year",
        help=f"Training data covers {ranges['Year'][0]:.0f}–{ranges['Year'][1]:.0f}.")
    present_price = col2.number_input(
        "Present showroom price of this model (₹ lakh)", min_value=0.1,
        max_value=500.0, value=8.0, step=0.1, format="%.2f", key="present_price",
        help="Current ex-showroom price of the same model when new. "
             "1 lakh = ₹100,000.")
    kms = col1.number_input(
        "Kilometres driven", min_value=0, max_value=1_000_000, value=40_000,
        step=1_000, key="kms")
    fuel = col2.selectbox("Fuel type", bundle["categories"]["Fuel_Type"],
                          index=bundle["categories"]["Fuel_Type"].index("Petrol"),
                          key="fuel")
    transmission = col1.selectbox(
        "Transmission", bundle["categories"]["Transmission"],
        index=bundle["categories"]["Transmission"].index("Manual"), key="transmission")
    submitted = st.form_submit_button("Estimate price")

# --- C. Result ------------------------------------------------------------
if submitted:
    result = predict_price(bundle, {
        "Year": year, "Present_Price": present_price, "Kms_Driven": kms,
        "Fuel_Type": fuel, "Transmission": transmission,
    })
    if result.errors:
        for msg in result.errors:
            st.error(msg)
    else:
        st.metric("Estimated resale price", f"₹ {result.price:.2f} lakh")
        st.caption(f"≈ ₹{result.price * 100_000:,.0f}. This is a model estimate, "
                   "not a guaranteed market price.")
        for msg in result.warnings:
            st.warning(msg)

st.info(
    "**Limitations:** see the test errors below for typical accuracy. "
    "Expensive models that hold their value (in the test set: Innova, Creta) tend to be "
    "**under**-estimated. The model does not know brand, condition, location or "
    "service history, and prices reflect the 2018–2020 market, not today's."
)

# --- D. Model information ---------------------------------------------------
st.header("About the model")
metrics = load_metrics()
if metrics is None:
    st.warning("reports/model_metrics.json not found; run `python -m src.train`.")
else:
    test = metrics["test"]["all"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"₹ {test['MAE']:.2f} lakh")
    c2.metric("Test RMSE", f"₹ {test['RMSE']:.2f} lakh")
    c3.metric("Test R²", f"{test['R2']:.3f}")
    st.caption(
        f"Held-out test set of {test['n']} cars "
        f"({metrics['split']['train']} used for training). MAE is the average "
        "size of an error. RMSE weights large errors more heavily. R² is the "
        "share of price variation explained."
    )
    st.write("**Model inputs:** manufacturing year (converted to age), present "
             "price, kilometres driven, fuel type, transmission.")
    st.write(f"**Algorithm:** {metrics['model']} on log price "
             f"(variant `{metrics['variant']}`).")
