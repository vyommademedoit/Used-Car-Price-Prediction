# Indian Used Car Resale Price Prediction Using Multiple Linear Regression

A college machine-learning project that estimates the resale price of a used
car in India from its age, kilometres driven, fuel type, transmission and
current showroom price, using **Multiple Linear Regression** (scikit-learn)
and a **Streamlit** web app.

> Academic prototype. Predictions are model estimates from 2018–2020
> listings and are not market valuations.

## 1. Problem statement
Given basic details of a used car, predict its selling price (₹ lakh). The
project covers the full workflow: data validation and cleaning, exploratory
analysis, a leakage-free regression pipeline, honest evaluation,
interpretation of the coefficients, and a simple prediction interface.

## 2. Objectives and scope
- Explore and clean a real Indian used-car dataset
- Build Multiple Linear Regression with scikit-learn `Pipeline` / `ColumnTransformer`
- Evaluate with MAE, RMSE and R² on a held-out test set
- Interpret the coefficients and the prediction errors
- Serve predictions through a Streamlit app

**Out of scope:** other algorithms (random forests, boosting, neural
networks), deployment infrastructure, databases, user accounts.

## 3. Dataset
| | |
|---|---|
| Name | *Vehicle dataset from CarDekho*, file `car data.csv` |
| Source | Kaggle, user nehalbirla: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho |
| Origin | Used-vehicle listings on CarDekho (India), around 2018–2020. Collection method not documented. |
| Licence | Database: ODbL v1.0; contents: DbCL v1.0 (as listed on Kaggle) |
| Size | 301 rows × 9 columns. After cleaning: **198 cars** |

| Column | Meaning | Used by model |
|---|---|---|
| `Selling_Price` | **Target**: resale price, ₹ lakh | target |
| `Present_Price` | Current ex-showroom price of the model, ₹ lakh | ✓ (log) |
| `Year` | Manufacturing year | ✓ (as age = 2020 − Year) |
| `Kms_Driven` | Odometer reading, km | ✓ |
| `Fuel_Type` | Petrol / Diesel / CNG | ✓ |
| `Transmission` | Manual / Automatic | ✓ |
| `Seller_Type` | Dealer / Individual | ✗ (only 5 individual sellers) |
| `Owner` | Number of previous owners | ✗ (only 5 cars with any) |
| `Car_Name` | Model name | ✗ (98 names, too few cars per name) |

Cleaning removed **101 two-wheelers** (motorcycles and scooters mixed
into the file) and **2 exact duplicates**. No missing or impossible
values were found. Details: [docs/dataset_assessment.md](docs/dataset_assessment.md).

## 4. Technology stack
Python 3.14 · pandas · NumPy · scikit-learn · Matplotlib / Seaborn ·
Streamlit · joblib · pytest. Exact versions are pinned in `requirements.txt`.

## 5. Setup

Needs **Python 3.11 or newer** (developed on 3.14). Open a terminal **inside the
project folder** (the one containing `app.py`) and run every command below from there.

```bash
python -m venv .venv
```

Activate the virtual environment:
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
  (if PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process Bypass` first)
- Windows (cmd): `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

If you skip activation, prefix commands with the venv's Python instead, e.g.
`.venv\Scripts\python -m src.train` (Windows) or `.venv/bin/python -m src.train`.

## 6. Get the dataset
The data is not stored in this repository.
1. Sign in to Kaggle and open the dataset page linked above.
2. Download `car data.csv`.
3. Save it as **`data/raw/car_data.csv`** (rename the file to remove the space).

If the file is missing, the scripts stop with a message explaining where
to put it.

## 7. Run the pipeline
Run these from the project root, in this order:

```bash
python -m src.data_preprocessing   # clean → data/processed/cars_clean.csv
python -m src.eda                  # figures 01–08 + reports/eda_summary.json
python -m src.train                # select, train, evaluate → models/ + reports/
```

`src.train` saves the fitted pipeline to `models/car_price_model.joblib`
and writes `reports/model_metrics.json`, `reports/coefficients.csv`,
`reports/test_predictions.csv` and figures 09–12. All randomness uses
seed 42. A fresh clone reproduces identical metrics.

## 8. Method and results

**Pipeline** (one scikit-learn object, so training and the app transform
inputs identically):
`Year → age` · `Present_Price → log` · one-hot `Fuel_Type`, `Transmission`
(reference levels Petrol, Manual) · `LinearRegression` on `log(Selling_Price)`
· `exp()` back to ₹ lakh.

**Leakage control:**
- The 80/20 split (158 / 40 cars, seed 42) happens before EDA or any
  fitting, and EDA uses only the training cars.
- The encoders are fitted inside the pipeline, on the training data only.
- The target scale was chosen by 5-fold × 10 repeated cross-validation on
  the training set, using a rule written down beforehand.
- The test set was evaluated once.

| Data | n | MAE (₹ lakh) | RMSE (₹ lakh) | R² |
|---|---|---|---|---|
| Training (cross-validated) | 158 | 0.708 | 0.962 | 0.953 |
| **Test, all cars** | **40** | **0.935** | **1.696** | **0.915** |
| Test, excluding 1 out-of-range car | 39 | 0.725 | 0.902 | 0.938 |
| Test, model *without* `Present_Price` | 40 | 2.603 | 5.476 | 0.111 |

**Coefficients** (each holds the other inputs fixed): −9.7 % price per
year of age; −2.8 % per 10,000 km; a car whose new price is 1 % higher
resells for about 0.82 % more; +21.1 % for diesel vs petrol; +5.6 % for
automatic vs manual. The CNG estimate (2 cars) is unreliable.

Full analysis: [docs/eda_findings.md](docs/eda_findings.md) and
[docs/model_results.md](docs/model_results.md). Every significant choice
and its reason is in [docs/decision_log.md](docs/decision_log.md).

## 9. Run the app

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. Enter the year, showroom price, km, fuel type
and transmission. The app shows the estimate in ₹ lakh, warns when
inputs are outside the training data, and lists the held-out test
metrics. It loads the saved model and never retrains.

## 10. Tests

```bash
python -m pytest
```

57 tests cover:
- schema validation and cleaning
- the split and the EDA diagnostics
- the pipeline recovering known coefficients
- save/load round trips
- invalid-input handling
- Streamlit app start-up and form prediction, using `AppTest`

Tests that need the real CSV or the trained model are skipped if those
files are missing.

## 11. Repository layout
```
app.py                    Streamlit interface
ui/charts.py              interactive Altair charts used by the app
assets/style.css          app styling (dark theme details)
.streamlit/config.toml    Streamlit theme (colours, fonts)
src/config.py             paths, seed, split size, reference year
src/data_preprocessing.py load, validate, clean, split
src/eda.py                exploratory figures and diagnostics (training set)
src/model.py              pipeline definition and coefficient table
src/train.py              CV selection, training, test evaluation, saving
src/evaluate.py           metrics, plots, coefficient interpretation
src/predict.py            input validation and prediction for the app
tests/                    pytest suite
docs/                     dataset assessment, EDA, results, decisions, report, viva
reports/                  metrics JSON, coefficient table, figures
data/raw, data/processed  dataset (not committed)
models/                   saved pipeline (not committed; created by src.train)
```

## 12. Limitations
- **Small sample:** 198 cars. Test metrics would move noticeably with a
  different split.
- **Expensive, popular models are systematically under-predicted.** All 5
  test cars that sold above 12 lakh were predicted too low.
- **Extrapolation:** showroom prices above 36.23 lakh, and ages or km
  outside the training range, are beyond the data. The app warns about
  these.
- **Few examples:** CNG (2 cars), individual sellers and multi-owner cars
  are too rare to model reliably.
- **Missing information:** no brand, condition, location or listing date,
  and prices are from about 2018–2020.
- **`Present_Price` dependency:** the user must look it up. Without it,
  accuracy drops sharply.

## 13. Possible future improvements (not implemented)
- A larger, more recent dataset with brand and city fields
- A brand or segment feature to capture model-specific value retention
- Prediction intervals instead of a single estimate
- Regularised linear models (Ridge) as a comparison, keeping MLR as the main model

## 14. Further documents
- [docs/report_outline.md](docs/report_outline.md): college report outline, filled with actual results
- [docs/viva_preparation.md](docs/viva_preparation.md): likely viva and interview questions with answers
