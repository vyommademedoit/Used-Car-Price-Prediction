"""Interactive Altair charts used by app.py (dark theme).

Every function returns an Altair chart; app.py shows it with st.altair_chart.
Colours: one fixed colour per category, checked for colour-blind separation
against the dark card background.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd

# Fixed category colours (same car type = same colour in every chart).
FUEL_COLORS = {"Petrol": "#3987e5", "Diesel": "#d95926", "CNG": "#199e70"}
TRANS_COLORS = {"Manual": "#3987e5", "Automatic": "#d95926"}
ACCENT = "#FF8A3D"
YOUR_CAR = "#FFC15E"
MUTED = "#8B93A7"
GRID = "#222939"
LOG_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]
NEG, POS = "#F87171", "#34D399"


def _style(chart: alt.Chart, height: int = 300) -> alt.Chart:
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID, domainColor="#2C3446", tickColor="#2C3446",
            labelColor=MUTED, titleColor="#B9C0D2", labelFont="Inter",
            titleFont="Inter", titleFontWeight=500, labelFontSize=11,
            titleFontSize=12, titlePadding=10)
        .configure_legend(
            labelColor="#B9C0D2", titleColor=MUTED, labelFont="Inter",
            titleFont="Inter", orient="top", symbolType="circle",
            titleFontWeight=500)
        .configure_title(color="#E7E9F0", font="Space Grotesk", fontSize=14,
                         anchor="start", fontWeight=600)
    )


def _color_scale(mapping: dict) -> alt.Scale:
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


# --- Charts about the user's car --------------------------------------------
def depreciation_curve(curve: pd.DataFrame, your_age: int, your_price: float) -> alt.Chart:
    """Predicted price as the same car gets older (other inputs fixed)."""
    base = alt.Chart(curve)
    area = base.mark_area(opacity=0.14, color=ACCENT, line=False).encode(
        x=alt.X("Age:Q", title="Car age (years)", scale=alt.Scale(nice=False)),
        y=alt.Y("Price:Q", title="Predicted price (₹ lakh)"))
    line = base.mark_line(color=ACCENT, strokeWidth=2.5).encode(
        x="Age:Q", y="Price:Q",
        tooltip=[alt.Tooltip("Year:O", title="Made in"),
                 alt.Tooltip("Age:Q", title="Age (years)"),
                 alt.Tooltip("Price:Q", title="Predicted ₹ lakh", format=".2f")])
    hover_pts = base.mark_point(size=90, opacity=0, color=ACCENT).encode(
        x="Age:Q", y="Price:Q",
        tooltip=[alt.Tooltip("Year:O", title="Made in"),
                 alt.Tooltip("Price:Q", title="Predicted ₹ lakh", format=".2f")])
    you = pd.DataFrame([{"Age": your_age, "Price": your_price, "Label": "Your car"}])
    you_pt = alt.Chart(you).mark_point(
        filled=True, size=220, color=YOUR_CAR, stroke="#0B0E15", strokeWidth=2.5
    ).encode(x="Age:Q", y="Price:Q",
             tooltip=[alt.Tooltip("Price:Q", title="Your estimate ₹ lakh", format=".2f")])
    you_lbl = alt.Chart(you).mark_text(
        align="left", dx=12, dy=-10, color="#E7E9F0", font="Inter", fontWeight=600
    ).encode(x="Age:Q", y="Price:Q", text="Label:N")
    return _style(area + line + hover_pts + you_pt + you_lbl, height=280)


def market_scatter(cars: pd.DataFrame, you: dict) -> alt.Chart:
    """Every car in the dataset: showroom price vs selling price, plus your car."""
    lo = 1.0
    hi = max(float(cars["Present_Price"].max()), float(you["Present_Price"])) * 1.1
    diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(
        strokeDash=[4, 4], color="#4B5468", strokeWidth=1.2).encode(x="x:Q", y="y:Q")
    diag_lbl = alt.Chart(pd.DataFrame({"x": [hi / 1.6], "y": [hi / 1.6],
                                       "t": ["no value lost"]})).mark_text(
        color=MUTED, angle=315, dy=-8, fontSize=10, font="Inter").encode(
        x="x:Q", y="y:Q", text="t:N")
    pts = alt.Chart(cars).mark_circle(size=60, opacity=0.75, stroke="#0B0E15",
                                      strokeWidth=1).encode(
        x=alt.X("Present_Price:Q", title="Showroom price when new (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[lo, hi], nice=False),
                axis=alt.Axis(values=LOG_TICKS, grid=True)),
        y=alt.Y("Selling_Price:Q", title="Actual selling price (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[0.3, hi], nice=False),
                axis=alt.Axis(values=LOG_TICKS)),
        color=alt.Color("Fuel_Type:N", title="Fuel", scale=_color_scale(FUEL_COLORS)),
        tooltip=[alt.Tooltip("Car_Name:N", title="Car"), alt.Tooltip("Year:O"),
                 alt.Tooltip("Present_Price:Q", title="Showroom ₹ lakh"),
                 alt.Tooltip("Selling_Price:Q", title="Sold for ₹ lakh"),
                 alt.Tooltip("Kms_Driven:Q", title="Km", format=",")])
    you_df = pd.DataFrame([{**you, "Label": "Your car"}])
    you_pt = alt.Chart(you_df).mark_point(
        shape="diamond", filled=True, size=320, color=YOUR_CAR, stroke="#0B0E15",
        strokeWidth=2).encode(
        x="Present_Price:Q", y="Price:Q",
        tooltip=[alt.Tooltip("Price:Q", title="Your estimate ₹ lakh", format=".2f")])
    you_lbl = alt.Chart(you_df).mark_text(
        align="left", dx=12, color="#E7E9F0", font="Inter", fontWeight=600).encode(
        x="Present_Price:Q", y="Price:Q", text="Label:N")
    return _style(diag + diag_lbl + pts + you_pt + you_lbl, height=280)


# --- Dataset exploration ------------------------------------------------------
def price_histogram(cars: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(cars).mark_bar(
        color="#3987e5", cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
        binSpacing=2).encode(
        x=alt.X("Selling_Price:Q", bin=alt.Bin(step=1), title="Selling price (₹ lakh)"),
        y=alt.Y("count():Q", title="Number of cars"),
        tooltip=[alt.Tooltip("Selling_Price:Q", bin=alt.Bin(step=1), title="Price band ₹ lakh"),
                 alt.Tooltip("count():Q", title="Cars")])
    return _style(chart, height=260)


def price_vs_age(cars: pd.DataFrame, colour_by: str) -> alt.Chart:
    mapping = FUEL_COLORS if colour_by == "Fuel_Type" else TRANS_COLORS
    title = "Fuel" if colour_by == "Fuel_Type" else "Transmission"
    chart = alt.Chart(cars).mark_circle(size=55, opacity=0.8, stroke="#0B0E15",
                                        strokeWidth=1).encode(
        x=alt.X("Age:Q", title="Car age (years)"),
        y=alt.Y("Selling_Price:Q", title="Selling price (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[0.3, 40], nice=False),
                axis=alt.Axis(values=LOG_TICKS)),
        color=alt.Color(f"{colour_by}:N", title=title, scale=_color_scale(mapping)),
        tooltip=[alt.Tooltip("Car_Name:N", title="Car"), alt.Tooltip("Year:O"),
                 alt.Tooltip("Selling_Price:Q", title="Sold for ₹ lakh"),
                 alt.Tooltip(f"{colour_by}:N", title=title)])
    return _style(chart, height=260)


def price_by_group(cars: pd.DataFrame, colour_by: str) -> alt.Chart:
    mapping = FUEL_COLORS if colour_by == "Fuel_Type" else TRANS_COLORS
    title = "Fuel" if colour_by == "Fuel_Type" else "Transmission"
    order = list(mapping)
    box = alt.Chart(cars).mark_boxplot(
        size=34, opacity=0.9, median={"color": "#E7E9F0", "strokeWidth": 2},
        rule={"color": "#4B5468"}, outliers={"size": 18, "opacity": 0.6}).encode(
        x=alt.X(f"{colour_by}:N", title=title, sort=order,
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Selling_Price:Q", title="Selling price (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[0.3, 40], nice=False),
                axis=alt.Axis(values=LOG_TICKS)),
        color=alt.Color(f"{colour_by}:N", scale=_color_scale(mapping), legend=None))
    counts = cars.groupby(colour_by).size().reset_index(name="n")
    counts["lbl"] = counts["n"].map(lambda n: f"{n} car{'s' if n != 1 else ''}")
    labels = alt.Chart(counts).mark_text(
        color=MUTED, font="Inter", fontSize=11, baseline="top").encode(
        x=alt.X(f"{colour_by}:N", sort=order), y=alt.value(4), text="lbl:N")
    return _style(box + labels, height=260)


# --- Model charts -------------------------------------------------------------
def coefficient_effects(coefs: dict) -> alt.Chart:
    """Percentage effect of each input, from the log-scale coefficients."""
    rows = [
        ("Each extra year of age", coefs.get("Car_Age"), 1),
        ("Each extra 10,000 km", coefs.get("Kms_Driven"), 10_000),
        ("Diesel instead of petrol", coefs.get("Fuel_Type_Diesel"), 1),
        ("CNG instead of petrol", coefs.get("Fuel_Type_CNG"), 1),
        ("Automatic instead of manual", coefs.get("Transmission_Automatic"), 1),
    ]
    df = pd.DataFrame([{"Input": name, "Effect": math.expm1(b * x) * 100}
                       for name, b, x in rows if b is not None])
    df["Direction"] = df["Effect"].map(lambda v: "Raises price" if v > 0 else "Lowers price")
    df["Label"] = df["Effect"].map(lambda v: f"{v:+.1f}%")
    order = df["Input"].tolist()
    bars = alt.Chart(df).mark_bar(cornerRadius=4, height=18).encode(
        y=alt.Y("Input:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=12, labelColor="#C9CFDD")),
        x=alt.X("Effect:Q", title="Change in predicted price (%)",
                scale=alt.Scale(domain=[-20, 25])),
        color=alt.Color("Direction:N", scale=alt.Scale(
            domain=["Lowers price", "Raises price"], range=[NEG, POS]), title=None),
        tooltip=[alt.Tooltip("Input:N"), alt.Tooltip("Label:N", title="Effect")])
    def labels(positive: bool) -> alt.Chart:
        return alt.Chart(df).transform_filter(
            alt.datum.Effect > 0 if positive else alt.datum.Effect <= 0
        ).mark_text(font="Inter", fontWeight=600, color="#E7E9F0", fontSize=12,
                    align="left" if positive else "right", dx=6 if positive else -6
        ).encode(y=alt.Y("Input:N", sort=order), x="Effect:Q", text="Label:N")

    text = labels(True) + labels(False)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#4B5468").encode(x="x:Q")
    return _style(zero + bars + text, height=230)


def cv_comparison(cv: dict) -> alt.Chart:
    names = {"raw": "Raw prices", "log_target": "Log price", "log_log": "Log–log (chosen)"}
    df = pd.DataFrame([{"Variant": names.get(k, k), "MAE": v["MAE_mean"],
                        "lo": v["MAE_mean"] - v["MAE_sd"], "hi": v["MAE_mean"] + v["MAE_sd"],
                        "R2": v["R2_mean"], "Chosen": k == "log_log"}
                       for k, v in cv.items()])
    order = df["Variant"].tolist()
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                  size=56).encode(
        x=alt.X("Variant:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("MAE:Q", title="Cross-validated MAE (₹ lakh) — lower is better"),
        color=alt.condition(alt.datum.Chosen, alt.value(ACCENT), alt.value("#3A4359")),
        tooltip=[alt.Tooltip("Variant:N"), alt.Tooltip("MAE:Q", title="Mean MAE", format=".3f"),
                 alt.Tooltip("R2:Q", title="Mean R²", format=".3f")])
    err = alt.Chart(df).mark_rule(color="#C9CFDD", strokeWidth=1.5).encode(
        x=alt.X("Variant:N", sort=order), y="lo:Q", y2="hi:Q")
    txt = alt.Chart(df).mark_text(dy=-10, font="Inter", fontWeight=600,
                                  color="#E7E9F0", baseline="bottom").encode(
        x=alt.X("Variant:N", sort=order), y="hi:Q", text=alt.Text("MAE:Q", format=".2f"))
    return _style(bars + err + txt, height=280)


def actual_vs_predicted(test: pd.DataFrame) -> alt.Chart:
    lo, hi = 1.0, float(max(test["Selling_Price"].max(), test["Predicted"].max())) * 1.15
    diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(
        strokeDash=[4, 4], color="#4B5468").encode(x="x:Q", y="y:Q")
    pts = alt.Chart(test).mark_circle(size=70, opacity=0.85, stroke="#0B0E15",
                                      strokeWidth=1).encode(
        x=alt.X("Selling_Price:Q", title="Actual price (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[lo, hi], nice=False),
                axis=alt.Axis(values=LOG_TICKS)),
        y=alt.Y("Predicted:Q", title="Predicted price (₹ lakh, log scale)",
                scale=alt.Scale(type="log", domain=[lo, hi], nice=False),
                axis=alt.Axis(values=LOG_TICKS)),
        color=alt.Color("Fuel_Type:N", title="Fuel", scale=_color_scale(FUEL_COLORS)),
        tooltip=[alt.Tooltip("Car_Name:N", title="Car"), alt.Tooltip("Year:O"),
                 alt.Tooltip("Selling_Price:Q", title="Actual ₹ lakh"),
                 alt.Tooltip("Predicted:Q", title="Predicted ₹ lakh"),
                 alt.Tooltip("Abs_Error:Q", title="Error ₹ lakh")])
    big = test.nlargest(3, "Abs_Error")
    lbl = alt.Chart(big).mark_text(align="left", dx=8, color=MUTED, font="Inter",
                                   fontSize=11).encode(
        x="Selling_Price:Q", y="Predicted:Q", text="Car_Name:N")
    return _style(diag + pts + lbl, height=300)


def residual_histogram(test: pd.DataFrame) -> alt.Chart:
    df = test.assign(Residual=test["Selling_Price"] - test["Predicted"])
    bars = alt.Chart(df).mark_bar(color="#3987e5", cornerRadiusTopLeft=3,
                                  cornerRadiusTopRight=3, binSpacing=2).encode(
        x=alt.X("Residual:Q", bin=alt.Bin(step=0.5),
                title="Actual − predicted (₹ lakh)  ·  right of 0 = model too low"),
        y=alt.Y("count():Q", title="Number of test cars"),
        tooltip=[alt.Tooltip("Residual:Q", bin=alt.Bin(step=0.5), title="Error band"),
                 alt.Tooltip("count():Q", title="Cars")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#E7E9F0", strokeDash=[3, 3]).encode(x="x:Q")
    return _style(bars + zero, height=300)
