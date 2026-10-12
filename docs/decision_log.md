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

## Open — Decide in Stage 3 on training data (cross-validation) only
1. **Target scale: raw vs log `Selling_Price`.** EDA evidence favours log.
   The price is skewed (2.38 vs −0.19 after log), and depreciation looks
   proportional (the share of the new price retained falls from 0.94 to
   about 0.22 with age). Keep plain `LinearRegression` either way. On the
   log scale, convert predictions back to lakhs before computing MAE,
   RMSE and R², so the two options are compared in the same units.
2. **Fuel reference level = Petrol**, not the alphabetical default (CNG).
   With CNG as the reference, the Diesel and Petrol dummies have VIF ≈ 17
   because only 2 cars are CNG.
3. **Near-constant features.** In training, `Seller_Type = Individual`
   has 5 cars, `Owner > 0` has 5 (`Owner = 3` is a single camry, the most
   influential point), and CNG has 2. The test set contains *none* of
   them, so their coefficients cannot be checked on held-out data. Decide
   whether to keep them on domain grounds, not on test scores.
4. **`Year` vs `Car_Age`.** Use `Car_Age` only. They carry the same
   information, so including both would make the design matrix exactly
   collinear.
