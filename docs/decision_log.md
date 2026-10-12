# Decision Log

Each entry records a choice that affects results, why it was made, and what
was considered instead.

## D1 — Use `car data.csv` from the CarDekho dataset (Stage 1)
**Decision:** Use the 301-row `car data.csv` from Kaggle's "Vehicle dataset
from CarDekho" (nehalbirla).
**Why:** It contains `Selling_Price` and `Present_Price` in ₹ lakhs, matching
the project brief and allowing the with/without `Present_Price` experiment.
**Alternative:** The 4,340-row `CAR DETAILS FROM CAR DEKHO.csv` has more rows
but no `Present_Price` and prices in rupees.
**Cost:** After cleaning only 198 rows remain, so test metrics will be noisy.

## D2 — Remove two-wheelers (Stage 1)
**Decision:** Remove the 101 motorcycles and scooters found in the file.
**Evidence:** Vehicle names identify them unambiguously (Bajaj, Hero, Royal
Enfield, Yamaha, TVS, etc.); the prefix list was cross-checked against an
independent naming pattern (car names are lower-case) and the two agree on
all 301 rows. Every two-wheeler is sold by an `Individual` seller, so mixing
them in made `Seller_Type` look strongly related to price (correlation of
"Individual" with price: −0.55 on all rows vs −0.03 on cars only).
**Why:** The project predicts *car* prices; two-wheelers would distort the
coefficients and the app would never be used for them.
**Reversible:** `clean_data(df, keep_two_wheelers=True)`.

## D3 — Remove exact duplicate rows (Stage 1)
**Decision:** Drop 2 rows that are identical in every column, including
kilometres driven.
**Why:** Identical km readings make genuinely separate listings unlikely,
and a duplicate split across train and test would inflate test scores.

## D4 — Keep high-priced and high-mileage cars (Stage 1)
**Decision:** Retain the land cruiser (₹35 lakh sale, ₹92.6 lakh present
price) and older high-km cars (up to 197,176 km).
**Why:** These are plausible real vehicles, not errors. Their influence on
the linear model will be examined in EDA and residual analysis rather than
removed in advance.

## D5 — Car age with a fixed reference year of 2020 (Stage 1)
**Decision:** `Car_Age = 2020 − Year`, set in `src/config.py`.
**Why:** Age is easier to interpret than calendar year. In a linear model the
reference year only shifts the intercept, so it does not change predictions.

## D6 — EDA on the training split only (Stage 2)
**Decision:** Split the 198 clean cars 80/20 (seed 42, `split_data()` in
`src/data_preprocessing.py`) *before* EDA, and draw every figure and
statistic from the 158 training cars. Stage 3 reuses the same function, so
it trains on exactly these rows.
**Why:** EDA findings drive modelling choices (target transform, encoding,
feature set). If test prices were explored, those choices would be
partly fitted to the test set and its score would be optimistic.
**Exception:** test-set *feature* ranges and category counts were checked
(no prices), to know in advance which predictions are extrapolations.
**Cost:** EDA sees 40 fewer cars. All the patterns found were already
visible on the full cleaned set in Stage 1, so nothing important is lost.

## D7 — Report test metrics with and without the out-of-range car (Stage 2)
**Decision:** The primary test metrics (MAE, RMSE, R²) will use **all 40**
test cars. A secondary line will report the same metrics on the 39 cars
inside the training feature ranges, excluding the land cruiser (present
price 92.6 vs training max 36.23).
**Why:** The land cruiser forces any linear model to extrapolate 2.5×
beyond its data, and one large error can dominate RMSE on 40 cars. This
rule is fixed now, before any model exists, so it cannot be chosen after
seeing which version looks better. The full-set number stays the
headline, so the result is not flattered.

## D8 — Model features (Stage 3, fixed before any test-set evaluation)
**Decision:** Inputs are `Year` (converted to `Car_Age` inside the
pipeline), `Present_Price`, `Kms_Driven`, `Fuel_Type` and `Transmission`.
`Seller_Type` and `Owner` are **excluded**.
**Why:** In training, only 5 cars are sold by individuals and only 5
have a previous owner. `Owner = 3` is a single camry, the most
influential training point. Their coefficients would mostly describe those
few cars, and the test set has none of them, so the coefficients could
never be checked. Showing them as inputs in the app would suggest a
reliability the model does not have. `Fuel_Type` stays in because
Diesel (45 cars) is well represented. The CNG coefficient (2 cars) is
reported but flagged as unreliable. `Year` and `Car_Age` are never used
together, because they carry the same information.
**Not used to decide:** any test-set number.

## D9 — Reference levels Petrol and Manual (Stage 3)
**Decision:** One-hot encode with Petrol and Manual as the dropped
reference levels, so each category coefficient means "compared with a
petrol manual car". Unseen levels at prediction time fall back to the
reference level with a warning; the app will block them before that.
**Why:** These are the most common levels. With CNG (2 cars) as the
reference, the fuel dummies had VIF ≈ 17 (see EDA).

## D10 — Target scale chosen by training-set cross-validation (Stage 3)
**Decision rule, fixed before running:** compare three pre-declared
`LinearRegression` variants with repeated 5-fold CV (10 repeats, seed 42)
on the 158 training cars, and pick the lowest mean CV RMSE in ₹ lakhs:
- `raw`: price ~ features
- `log_target`: log(price) ~ features, predictions converted back with exp
- `log_log`: log(price) ~ log(Present_Price) + other features

**Why:** EDA showed a skewed target and percentage-style depreciation, but
the choice should come from data the model is allowed to see. All three
variants are still Multiple Linear Regression; only the scale of the
inputs and outputs changes. The no-`Present_Price` experiment uses the
chosen target scale.
**Known caveat:** exp(predicted log price) estimates the *median* price
for those inputs, which is slightly below the mean. This is acceptable for
a "typical price" estimate.

## Resolved items (formerly "Open")
Open items 1–4 from Stage 2 are resolved by D8, D9 and D10.
