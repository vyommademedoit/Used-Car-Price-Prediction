# College Report Outline

**Title:** Indian Used Vehicle Resale Price Prediction Using Multiple Linear Regression

All numbers come from the project's own outputs (`reports/*.json`,
`reports/coefficients.csv`). Sections marked *(write)* need the student's
own prose. The bullets give the content to cover.

---

## Abstract *(write, ~200 words)*
- **Problem:** estimate used-car resale prices in India from basic
  listing details.
- **Data:** CarDekho `car data.csv` (Kaggle): 301 rows, reduced to 198 cars
  after removing 101 two-wheelers and 2 duplicates.
- **Method:** Multiple Linear Regression on log selling price.
  - Inputs: car age, km driven, log present price, fuel type, transmission.
  - 80/20 split (seed 42).
  - Target scale chosen by repeated 5-fold cross-validation on the
    training set.
- **Results on the 40-car test set:** MAE ₹0.935 lakh, RMSE ₹1.696 lakh,
  R² 0.915.
  - Without the present-price feature, R² falls to 0.111.
- **Delivered as:** a Streamlit app with input validation.
- **Main limitations:** small sample, and under-prediction of expensive
  models.

## 1. Introduction *(write)*
- India's used-car market. Buyers and sellers need a quick, explainable
  price estimate.
- Why an interpretable model (linear regression) suits a first, explainable
  estimate.
- Report structure.

## 2. Problem Statement
Develop a machine-learning application that predicts a used car's resale
price (₹ lakh) from its manufacturing year, kilometres driven, fuel type,
transmission and current showroom price, using Multiple Linear Regression.

## 3. Objectives
1. Validate, clean and explore a real Indian used-car dataset.
2. Build a leakage-free preprocessing + regression pipeline.
3. Evaluate with MAE, RMSE, R² on held-out data.
4. Interpret coefficients and errors.
5. Deploy as an interactive Streamlit application.

## 4. Dataset Description
- Source, licence (ODbL/DbCL), and the columns and units table: see
  README §3 and `docs/dataset_assessment.md`.
- **Data-quality findings:**
  - 101 motorcycles and scooters were mixed into the "car" data. Their
    presence created a fake seller-type effect: the correlation between
    "individual seller" and price was −0.55 with them and −0.03 without.
  - 2 duplicate rows.
  - No missing values.
- **Clean data:** 198 cars made in 2003–2018, priced 0.35–35 lakh (median
  5.25). Categories: Petrol 138, Diesel 58, CNG 2; Manual 168, Automatic 30.
- **Table:** cleaning steps and row counts, from
  `reports/data_cleaning_report.json`.

## 5. Methodology
### 5.1 Workflow
Raw CSV → validation → cleaning → **train/test split** → EDA (train
only) → candidate pipelines → cross-validation (train only) → final fit →
single test evaluation → saved pipeline → Streamlit app.

### 5.2 Exploratory analysis (figures 01–08)
Include the figures with one-paragraph interpretations from
`docs/eda_findings.md`:
- right-skewed price (skew 2.38; −0.19 after log)
- share of value retained falls with age (0.94 at 2 years, 0.45 at 10;
  r = −0.87), i.e. percentage-style depreciation
- the diesel and automatic price premiums mostly reflect pricier models
- correlation of age and km = 0.71
- influential points (2006 Camry, Cook's D 1.03)
- the out-of-range land cruiser

### 5.3 Feature engineering
- `Car_Age = 2020 − Year`. Year and age are never used together (they are
  perfectly collinear).
- `log(Present_Price)`
- One-hot encoding with Petrol and Manual as the reference levels. Using
  CNG as the reference gave a VIF of about 17.
- Excluded: `Seller_Type` and `Owner` (5 minority cars each, none in
  test) and `Car_Name`.
- No scaling: ordinary least squares does not need it, and unscaled
  inputs keep the coefficients in real units.

### 5.4 Leakage prevention
- Split before EDA.
- Encoders fitted inside the pipeline, on training data only.
- Model selection by CV on training data, with the rule fixed in advance
  (decision log D10).
- The test set was used once.
- `Present_Price` is a property of the model, available before a sale,
  and is not derived from the target.

### 5.5 Model selection
Table of the three candidate variants and their CV results (figure 09,
`docs/model_results.md` §2): raw 1.815, log-target 2.107, **log-log 0.962**
CV RMSE.

## 6. Mathematical Background
- **Multiple Linear Regression:**
  ŷ = b₀ + b₁x₁ + … + bₙxₙ. Least squares chooses b to minimise
  Σ(yᵢ − ŷᵢ)². The closed-form solution is b = (XᵀX)⁻¹Xᵀy.
- **The model used:**
  log(price) = b₀ + b₁·age + b₂·km + b₃·log(present) + b₄·CNG + b₅·Diesel + b₆·Automatic
  - It is still linear in the coefficients.
  - The effect of a dummy or unit change is e^b − 1 (a percentage).
  - b₃ is an elasticity.
- **One-hot encoding** and the dummy-variable trap: why one reference
  level is dropped.
- **Metrics:**
  - MAE = mean|y − ŷ|
  - RMSE = √mean(y − ŷ)²
  - R² = 1 − SS_res/SS_tot
- **Diagnostics:** VIF = 1/(1 − R²ⱼ). Cook's distance.
- **Assumptions:** linearity, independent errors, constant variance,
  roughly normal residuals. Explain how the log scale helps with the
  first and third.

## 7. System Architecture
- **Diagram:** data/raw → `src.data_preprocessing` → `src.eda` /
  `src.train` (uses `src.model`, `src.evaluate`) → `models/*.joblib` +
  `reports/` → `src.predict` → `app.py` (Streamlit).
- A single scikit-learn pipeline object guarantees identical
  preprocessing in training and in the app.

## 8. Implementation
- Tech stack and versions (`requirements.txt`).
- Short code excerpts:
  - `build_pipeline()` in `src/model.py`
  - the cross-validation loop in `src/train.py`
  - input validation in `src/predict.py`
- Testing: 57 pytest tests (cleaning, pipeline, save/load, invalid
  inputs, Streamlit `AppTest`).
- App screenshots *(take them yourself after running `streamlit run app.py`)*.

## 9. Results and Discussion
- **Metrics table** (README §8): test MAE 0.935, RMSE 1.696, R² 0.915.
  Train CV RMSE 0.962. No overfitting gap.
- **Explaining the metrics:**
  - Average error ≈ ₹93,500.
  - Median error 15 % of price.
  - 31 of 40 test cars within ₹1 lakh.
  - RMSE is inflated by one car (the land cruiser, error 9.12 lakh).
- **Figures:** 10 (actual vs predicted), 11 and 12 (residuals).
- **Coefficient table** with plain-language effects (`docs/model_results.md` §6).
- **Present_Price experiment:** without it, test R² drops from 0.915 to
  0.111. The showroom price captures car class, which no other input
  does.
- **Error analysis:** all 5 test cars that sold above 12 lakh are
  under-predicted, while cheaper cars are slightly over-predicted on
  average.

## 10. Limitations
Copy from README §12. Be explicit about the small sample, the 2018–2020
market, extrapolation, rare categories and the missing brand feature.

## 11. Conclusion and Future Scope
- **Conclusion *(write)*:** a simple, interpretable MLR model explains
  most of the price variation on these mass-market cars.
  - It depends heavily on the showroom price.
  - It reveals roughly 10 %-per-year depreciation.
- **Future scope (proposed, not done):**
  - a larger, more recent dataset with brand and city fields
  - a brand or segment feature
  - prediction intervals
  - Ridge regression as a comparison

## 12. References *(format in your college's style)*
1. N. Birla, "Vehicle dataset from CarDekho", Kaggle.
   https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12, 2011.
3. G. James, D. Witten, T. Hastie, R. Tibshirani, *An Introduction to
   Statistical Learning*, Springer — chapter on linear regression.
4. Streamlit documentation. https://docs.streamlit.io
5. Open Data Commons, ODbL v1.0 and DbCL v1.0 licence texts.
