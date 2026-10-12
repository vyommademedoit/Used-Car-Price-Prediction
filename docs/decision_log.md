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

## Open — Near-constant features (decide in Stage 3)
After cleaning, `Seller_Type` (193 Dealer / 5 Individual), `Owner`
(193 / 4 / 1) and `Fuel_Type = CNG` (2 rows) have very few non-majority
examples. Their coefficients will rest on a handful of cars. This will be
decided on domain grounds and training data only, never by checking test
scores.
