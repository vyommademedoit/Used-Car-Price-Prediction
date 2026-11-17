"""Streamlit app: estimate a used car's resale price.

Run from the project root (after `python -m src.train`):
    streamlit run app.py

The app only loads the saved pipeline; it never retrains.
Look and feel: .streamlit/config.toml (dark theme), assets/style.css,
and the interactive charts in ui/charts.py.
"""

import csv
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src.predict import ModelNotFoundError, load_model_bundle, predict_price
from ui import charts

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
INPUT_KEYS = ("year", "kms", "present_price", "fuel", "transmission")
DEFAULTS = {"year": 2015, "kms": 40_000, "present_price": 8.0,
            "fuel": "Petrol", "transmission": "Manual"}

# Real cars from the held-out test set (the model never saw them in training).
EXAMPLES = {
    "Hyundai i20 · 2016": dict(year=2016, kms=7_000, present_price=7.6,
                               fuel="Petrol", transmission="Manual", actual=6.25),
    "Honda City · 2014": dict(year=2014, kms=48_000, present_price=9.9,
                              fuel="Diesel", transmission="Manual", actual=7.2),
    "Hyundai Creta · 2016": dict(year=2016, kms=35_934, present_price=13.6,
                                 fuel="Diesel", transmission="Manual", actual=12.9),
    "Toyota Innova AT · 2017": dict(year=2017, kms=11_000, present_price=23.15,
                                    fuel="Petrol", transmission="Automatic", actual=19.75),
}

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗",
                   layout="wide", initial_sidebar_state="collapsed")


# --- Loading ---------------------------------------------------------------
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


def load_coefficients() -> dict[str, float]:
    """Coefficients saved by training, e.g. {"Car_Age": -0.1016, ...}."""
    path = config.REPORTS_DIR / "coefficients.csv"
    try:
        with path.open(encoding="utf-8") as f:
            return {row["term"]: float(row["coefficient"]) for row in csv.DictReader(f)}
    except (OSError, KeyError, ValueError):
        return {}


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except OSError:
        return None


# --- Small helpers -----------------------------------------------------------
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def show_chart(chart) -> None:
    st.altair_chart(chart, theme=None, width="stretch")


def rupees_indian(amount: float) -> str:
    """Format rupees with Indian digit grouping: 532000 -> '5,32,000'."""
    s = f"{round(amount):d}"
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def to_model_inputs(values: dict) -> dict:
    return {"Year": values["year"], "Present_Price": values["present_price"],
            "Kms_Driven": values["kms"], "Fuel_Type": values["fuel"],
            "Transmission": values["transmission"]}


def factor_rows(coefs: dict, inputs: dict) -> list[tuple[str, str, float | None]]:
    """Effect of each input on its own, as a % change in price.

    The model works on log(price), so an input x with coefficient b
    multiplies the price by exp(b·x). The % change is exp(b·x) − 1,
    compared with a 0-year-old, 0-km, petrol, manual car.
    """
    age = config.REFERENCE_YEAR - int(inputs["Year"])
    kms = float(inputs["Kms_Driven"])
    fuel, trans = inputs["Fuel_Type"], inputs["Transmission"]

    def pct(term: str, x: float) -> float | None:
        return math.expm1(coefs[term] * x) * 100 if term in coefs else None

    rows = [
        ("Age", f"{age} year{'s' if age != 1 else ''} old", pct("Car_Age", age)),
        ("Kilometres", f"{kms:,.0f} km driven", pct("Kms_Driven", kms)),
    ]
    if fuel == "Petrol":
        rows.append(("Fuel", "Petrol (baseline)", 0.0))
    else:
        rows.append(("Fuel", f"{fuel} vs Petrol", pct(f"Fuel_Type_{fuel}", 1)))
    if trans == "Manual":
        rows.append(("Transmission", "Manual (baseline)", 0.0))
    else:
        rows.append(("Transmission", f"{trans} vs Manual", pct(f"Transmission_{trans}", 1)))
    return rows


def factor_html(rows) -> str:
    shown = [r for r in rows if r[2] is not None]
    biggest = max([abs(r[2]) for r in shown] + [1.0])
    out = []
    for name, detail, value in shown:
        width = abs(value) / biggest * 50  # half the bar each side of the centre line
        kind = "zero" if abs(value) < 0.05 else ("neg" if value < 0 else "pos")
        seg = f'<div class="seg {kind}" style="width:{width:.1f}%"></div>' if kind != "zero" else ""
        pct = "base" if kind == "zero" else f"{value:+.1f}%"
        out.append(
            f'<div class="factor"><div class="name">{name}<small>{detail}</small></div>'
            f'<div class="bar">{seg}</div><div class="pct {kind}">{pct}</div></div>')
    return "".join(out)


def depreciation_data(bundle: dict, inputs: dict) -> pd.DataFrame:
    """Predicted price for the same car built in each year the data covers."""
    lo, hi = (int(v) for v in bundle["training_ranges"]["Year"])
    rows = []
    for year in range(lo, hi + 1):
        p = predict_price(bundle, {**inputs, "Year": year}).price
        if p is not None:
            rows.append({"Year": year, "Age": config.REFERENCE_YEAR - year, "Price": p})
    return pd.DataFrame(rows)


def section_head(kicker: str, title: str, sub: str) -> None:
    html(f'<div class="section-head"><div class="k">{kicker}</div><h2>{title}</h2>'
         f"<p>{sub}</p></div>")


# --- Session state & callbacks ----------------------------------------------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("last_inputs", None)
st.session_state.setdefault("actual_price", None)


def on_submit() -> None:
    st.session_state.last_inputs = {k: st.session_state[k] for k in INPUT_KEYS}
    st.session_state.actual_price = None
    st.session_state.example = None


def on_example() -> None:
    name = st.session_state.get("example")
    if not name:
        return
    ex = EXAMPLES[name]
    for k in INPUT_KEYS:
        st.session_state[k] = ex[k]
    st.session_state.last_inputs = {k: ex[k] for k in INPUT_KEYS}
    st.session_state.actual_price = ex["actual"]


# --- A. Introduction ---------------------------------------------------------
css_path = ASSETS_DIR / "style.css"
if css_path.exists():
    html(f"<style>{css_path.read_text(encoding='utf-8')}</style>")

metrics = load_metrics()
coefs = load_coefficients()
cars = load_csv(str(config.PROCESSED_DATA_PATH))
test_preds = load_csv(str(config.REPORTS_DIR / "test_predictions.csv"))

html('<span class="eyebrow"><span class="dot"></span>Multiple Linear Regression · CarDekho listings</span>')
st.title("Indian Used Car Resale Price Estimator")
html(
    '<p class="hero-sub">Estimates what a used car will sell for from its <b>age</b>, '
    "<b>kilometres driven</b>, <b>fuel type</b>, <b>transmission</b> and current "
    "<b>showroom price</b>. Built from 198 CarDekho car listings from around 2018–2020 "
    "(158 used for training, 40 held back for testing).</p>"
)
if metrics:
    t = metrics["test"]["all"]
    html(
        '<div class="chips">'
        f'<div class="chip"><div class="k">Cars in dataset</div><div class="v">{metrics["split"]["train"] + metrics["split"]["test"]}</div></div>'
        f'<div class="chip"><div class="k">Test R²</div><div class="v">{t["R2"]:.3f}</div></div>'
        f'<div class="chip"><div class="k">Avg. error (MAE)</div><div class="v">₹ {t["MAE"]:.2f} lakh</div></div>'
        '<div class="chip"><div class="k">Model</div><div class="v">Linear · log–log</div></div>'
        "</div>"
    )

try:
    bundle = get_bundle()
except (ModelNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

ranges = bundle["training_ranges"]

# --- B. Prediction form + C. Result -----------------------------------------
form_col, result_col = st.columns([5, 6], gap="large")

with form_col:
    with st.container(key="examples_box"):
        st.pills("Try a real car the model never saw", list(EXAMPLES), key="example",
                 on_change=on_example,
                 help="These cars were held back for testing. Pick one to fill the "
                      "form and compare the estimate with the real selling price.")

    with st.form("car_form", border=True):
        html('<div class="card-title">Car details</div>'
             '<div class="card-sub">Or enter your own car below.</div>')

        html('<div class="step"><span class="n">1</span>Age &amp; usage</div>')
        c1, c2 = st.columns(2)
        c1.number_input(
            "Manufacturing year", min_value=1990, max_value=config.REFERENCE_YEAR,
            step=1, key="year",
            help=f"Training data covers {ranges['Year'][0]:.0f}–{ranges['Year'][1]:.0f}. "
                 f"Age is counted from {config.REFERENCE_YEAR}.")
        c2.number_input(
            "Kilometres driven", min_value=0, max_value=1_000_000, step=5_000, key="kms",
            help=f"Training data covers {ranges['Kms_Driven'][0]:,.0f}–"
                 f"{ranges['Kms_Driven'][1]:,.0f} km.")

        html('<div class="step"><span class="n">2</span>Price when new</div>')
        st.number_input(
            "Present showroom price of this model (₹ lakh)", min_value=0.1,
            max_value=500.0, step=0.5, format="%.2f", key="present_price",
            help="Current ex-showroom price of the same model when new. "
                 "1 lakh = ₹100,000.")
        html(f'<div class="hint">Most cars in the data cost ₹ {ranges["Present_Price"][0]:g}–'
             f'{ranges["Present_Price"][1]:g} lakh new · 1 lakh = ₹1,00,000</div>')

        html('<div class="step"><span class="n">3</span>Specs</div>')
        st.segmented_control("Fuel type", bundle["categories"]["Fuel_Type"][::-1],
                             key="fuel", selection_mode="single")
        st.segmented_control("Transmission", bundle["categories"]["Transmission"][::-1],
                             key="transmission", selection_mode="single")
        if "CNG" in bundle["categories"]["Fuel_Type"]:
            html('<div class="hint">Only 2 CNG cars were in the data, so CNG estimates '
                 "are rough.</div>")

        st.form_submit_button("Estimate price", type="primary", width="stretch",
                              on_click=on_submit)

last = st.session_state.last_inputs
inputs = to_model_inputs(last) if last else None
result = predict_price(bundle, inputs) if inputs else None
ok = result is not None and not result.errors

with result_col:
    with st.container(border=True, key="result_box"):
        if result is None:
            html(
                '<div class="placeholder">'
                '<svg width="88" height="52" viewBox="0 0 88 52" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M8 34c0-3 1.6-5 4.5-6l9-3.2 8.6-10.2c1.5-1.8 3.7-2.8 6-2.8h17.4c2.4 0 4.6 1.1 6.1 3l7.6 9.6'
                ' 9.8 2.6c3.4.9 5 3.2 5 6.6V38c0 1.7-1.3 3-3 3h-5" stroke="#FF8A3D" stroke-width="2.4" stroke-linecap="round"/>'
                '<path d="M8 34v4c0 1.7 1.3 3 3 3h4M29 41h29" stroke="#FF8A3D" stroke-width="2.4" stroke-linecap="round"/>'
                '<circle cx="22" cy="41" r="6.5" stroke="#FFC15E" stroke-width="2.4"/>'
                '<circle cx="65" cy="41" r="6.5" stroke="#FFC15E" stroke-width="2.4"/></svg>'
                '<div class="t">Your estimate will appear here</div>'
                '<div class="s">Pick one of the example cars, or fill in the form and press '
                "<b>Estimate price</b>. You’ll see the predicted value, which inputs pushed it "
                "up or down, and charts of how it compares.</div></div>"
            )
        elif result.errors:
            for msg in result.errors:
                st.error(msg)
        else:
            price = result.price
            present_price = float(inputs["Present_Price"])
            st.metric("Estimated resale price", f"₹ {price:.2f} lakh")
            html(f'<div class="price-words">≈ <b>₹{rupees_indian(price * 100_000)}</b> · '
                 "a model estimate, not a guaranteed market price.</div>")

            actual = st.session_state.actual_price
            if actual is not None:
                diff = price - actual
                html(f'<div class="actual"><span>Real selling price of this car</span>'
                     f"<b>₹ {actual:.2f} lakh</b><em>model was {abs(diff):.2f} lakh "
                     f"{'too high' if diff > 0 else 'too low'} ({abs(diff) / actual:.0%})</em></div>")

            retained = max(0.0, min(price / present_price * 100, 100.0))
            html(
                '<div class="range"><div class="row"><span>Value kept vs. new showroom price</span>'
                f"<span><b style='color:#E7E9F0'>{retained:.0f}%</b> of ₹ {present_price:.2f} lakh</span></div>"
                f'<div class="track"><div class="fill" style="width:{retained:.1f}%"></div></div></div>'
            )
            if metrics and not result.warnings:
                mae = metrics["test"]["all"]["MAE"]
                lo, hi = max(price - mae, 0), price + mae
                html(f'<div class="range"><div class="note">On the 40 test cars the average error was '
                     f"₹ {mae:.2f} lakh, so a realistic band is roughly "
                     f"<b style='color:#E7E9F0'>₹ {lo:.2f} – {hi:.2f} lakh</b>.</div></div>")

            for msg in result.warnings:
                st.warning(msg)

            if coefs:
                html('<div class="section-mini">What moved the price</div>'
                     '<div class="section-mini-sub">Effect of each input on its own, compared with a '
                     "brand-new petrol manual car of the same showroom price.</div>")
                html(factor_html(factor_rows(coefs, inputs)))
                if "log_Present_Price" in coefs:
                    html(f'<div class="anchor-note">The <b>showroom price</b> sets the starting point: '
                         f"every 1% higher showroom price gives about "
                         f"{coefs['log_Present_Price']:.2f}% higher resale price.</div>")

# --- Your car in context -------------------------------------------------------
if ok:
    section_head("Your car in context", "How this estimate compares",
                 "Charts update every time you estimate a new car. Hover for details.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="chart_dep"):
            html('<div class="chart-title">Value as the car gets older</div>'
                 '<div class="chart-sub">Same car, same km and showroom price, '
                 "built in different years.</div>")
            curve = depreciation_data(bundle, inputs)
            show_chart(charts.depreciation_curve(
                curve, config.REFERENCE_YEAR - int(inputs["Year"]), result.price))
    with right:
        with st.container(border=True, key="chart_market"):
            html('<div class="chart-title">Your car among all 198 listings</div>'
                 '<div class="chart-sub">Further below the dashed line = more value lost.</div>')
            if cars is not None:
                show_chart(charts.market_scatter(
                    cars, {"Present_Price": float(inputs["Present_Price"]),
                           "Price": result.price}))

html(
    '<div class="limits"><div class="ic">⚠️</div><div><b>Limitations.</b> Expensive models '
    "that hold their value (in the test set: Innova, Creta) tend to be <b>under</b>-estimated. "
    "The model does not know brand, condition, location or service history, and prices "
    "reflect the 2018–2020 market, not today’s.</div></div>"
)

# --- Explore the data ----------------------------------------------------------
if cars is not None:
    section_head("The data", "Explore the dataset",
                 f"{len(cars)} cleaned CarDekho listings used to build the model.")
    data = cars.assign(Age=config.REFERENCE_YEAR - cars["Year"])
    colour_by = st.segmented_control(
        "Group cars by", ["Fuel type", "Transmission"], default="Fuel type",
        key="colour_by", label_visibility="collapsed") or "Fuel type"
    col = "Fuel_Type" if colour_by == "Fuel type" else "Transmission"
    a, b, c = st.columns(3, gap="medium")
    for box, key, title, sub, chart in [
        (a, "chart_hist", "How prices are spread", "Most used cars sell for under ₹ 10 lakh.",
         charts.price_histogram(data)),
        (b, "chart_age", "Older cars sell for less", "Each dot is one listing.",
         charts.price_vs_age(data, col)),
        (c, "chart_group", f"Price by {colour_by.lower()}",
         "Box = middle half of cars, line = median.", charts.price_by_group(data, col)),
    ]:
        with box, st.container(border=True, key=key):
            html(f'<div class="chart-title">{title}</div><div class="chart-sub">{sub}</div>')
            show_chart(chart)

# --- D. Model information ----------------------------------------------------
section_head("Under the hood", "About the model",
             "How the model was built, and how well it does on cars it never saw.")

html(
    '<div class="pipeline">'
    '<div class="node"><div class="ic">📄</div><b>Raw data</b><span>301 CarDekho rows</span></div>'
    '<div class="arrow">→</div>'
    '<div class="node"><div class="ic">🧹</div><b>Cleaning</b><span>−101 two-wheelers, −2 duplicates → 198 cars</span></div>'
    '<div class="arrow">→</div>'
    '<div class="node"><div class="ic">🛠️</div><b>Features</b><span>age, km, log(showroom price), fuel, gearbox</span></div>'
    '<div class="arrow">→</div>'
    '<div class="node"><div class="ic">✂️</div><b>Split</b><span>158 train · 40 test</span></div>'
    '<div class="arrow">→</div>'
    '<div class="node hl"><div class="ic">📈</div><b>Linear regression</b><span>fits log(price) by least squares</span></div>'
    '<div class="arrow">→</div>'
    '<div class="node"><div class="ic">🚗</div><b>Estimate</b><span>exp(·) → ₹ lakh</span></div>'
    "</div>"
)

if metrics is None:
    st.warning("reports/model_metrics.json not found; run `python -m src.train`.")
else:
    test = metrics["test"]["all"]
    with st.container(key="metrics_row"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Test MAE", f"₹ {test['MAE']:.2f} lakh")
        m2.metric("Test RMSE", f"₹ {test['RMSE']:.2f} lakh")
        m3.metric("Test R²", f"{test['R2']:.3f}")
    st.caption(
        f"Held-out test set of {test['n']} cars "
        f"({metrics['split']['train']} used for training). MAE is the average "
        "size of an error. RMSE weights large errors more heavily. R² is the "
        "share of price variation explained."
    )

    tab_how, tab_compare, tab_diag = st.tabs(
        ["How it works", "Why log–log?", "Test diagnostics"])

    with tab_how:
        left, right = st.columns([5, 6], gap="large")
        with left:
            st.markdown("The model predicts the **logarithm** of the selling price "
                        "as a weighted sum of the inputs:")
            st.latex(r"\begin{aligned}\log(\text{price}) = {} & b_0 + b_1\,\text{age} "
                     r"+ b_2\,\text{km} \\ & + b_3\log(\text{present price}) \\ "
                     r"& + b_4\,\text{diesel} + b_5\,\text{CNG} + b_6\,\text{automatic}"
                     r"\end{aligned}")
            st.markdown(
                "Each weight *b* was learned from the training cars by **least squares**, "
                "which picks the weights that minimise the squared prediction errors. "
                "Working on the log scale means each input changes the price by a "
                "*percentage*.  \n"
                f"**Algorithm:** {metrics['model']} on log price "
                f"(variant `{metrics['variant']}`).")
        with right:
            if coefs:
                with st.container(border=True, key="chart_coef"):
                    html('<div class="chart-title">What each input does to the price</div>'
                         '<div class="chart-sub">From the learned coefficients, other inputs '
                         "held equal. Showroom price: +1% → "
                         f"+{coefs.get('log_Present_Price', 0):.2f}%.</div>")
                    show_chart(charts.coefficient_effects(coefs))
        coef_path = config.REPORTS_DIR / "coefficients.csv"
        if coef_path.exists():
            with st.expander("Show the raw coefficient table"):
                table = pd.read_csv(coef_path)
                table["coefficient"] = table["coefficient"].map(lambda c: f"{c:.4g}")
                st.dataframe(
                    table.rename(columns={"term": "Term", "coefficient": "Coefficient",
                                          "effect": "What it means"}),
                    hide_index=True, width="stretch",
                    column_config={"What it means": st.column_config.TextColumn(width="large")})

    with tab_compare:
        left, right = st.columns([5, 6], gap="large")
        with left:
            st.markdown(
                "Three versions of the regression were compared with repeated 5-fold "
                "cross-validation on the training set. Taking the log of both the "
                "selling price and the present price (**log–log**) gave the lowest "
                "errors, so it was chosen.")
            cv = metrics.get("cv_candidates", {})
            if cv:
                names = {"raw": "Raw prices", "log_target": "Log price",
                         "log_log": "Log–log ✓"}
                st.dataframe(
                    pd.DataFrame([{"Variant": names.get(k, k), "MAE (₹ lakh)": v["MAE_mean"],
                                   "RMSE (₹ lakh)": v["RMSE_mean"], "R²": v["R2_mean"]}
                                  for k, v in cv.items()]),
                    hide_index=True, width="stretch")
                st.caption(metrics.get("cv_scheme", ""))
        with right:
            if metrics.get("cv_candidates"):
                with st.container(border=True, key="chart_cv"):
                    html('<div class="chart-title">Cross-validation error by model version</div>'
                         '<div class="chart-sub">Bar = mean MAE, line = ± one standard deviation.</div>')
                    show_chart(charts.cv_comparison(metrics["cv_candidates"]))

    with tab_diag:
        if test_preds is None:
            st.caption("reports/test_predictions.csv not found; run `python -m src.train`.")
        else:
            left, right = st.columns(2, gap="large")
            with left, st.container(border=True, key="chart_avp"):
                html('<div class="chart-title">Actual vs predicted (40 test cars)</div>'
                     '<div class="chart-sub">Dots on the dashed line were predicted perfectly.</div>')
                show_chart(charts.actual_vs_predicted(test_preds))
            with right, st.container(border=True, key="chart_resid"):
                html('<div class="chart-title">Size of the errors</div>'
                     '<div class="chart-sub">Most errors are within ±₹ 1 lakh; the far-right '
                     "bar is the Land Cruiser, priced far above the training range.</div>")
                show_chart(charts.residual_histogram(test_preds))

html('<div class="footer"><span>Used Car Price Prediction · Multiple Linear Regression</span>'
     "<span>Data: CarDekho listings (≈2018–2020) · Prices in ₹ lakh</span></div>")
