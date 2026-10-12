# Dataset Assessment

## Source
- **Dataset:** "Vehicle dataset from CarDekho", Kaggle user nehalbirla
  (https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
- **File used:** `car data.csv`, saved locally as `data/raw/car_data.csv`
- **Origin:** Used-vehicle listings from the Indian site CarDekho, as stated by
  the uploader. Collection method and date are not documented.
- **Licence:** Not yet confirmed. Check the Kaggle page before redistributing.
- **Download:** Manual, through Kaggle (requires a login).

## Raw file
301 rows × 9 columns, no missing values.

| Column | Type | Meaning / unit |
|---|---|---|
| Car_Name | text | Model name (e.g. `city`, `fortuner`) |
| Year | integer | Year of manufacture (2003–2018) |
| Selling_Price | number | **Target.** Resale price, ₹ lakhs |
| Present_Price | number | Current ex-showroom price of the model, ₹ lakhs |
| Kms_Driven | integer | Odometer reading, km |
| Fuel_Type | category | Petrol, Diesel, CNG |
| Seller_Type | category | Dealer, Individual |
| Transmission | category | Manual, Automatic |
| Owner | integer | Number of previous owners (0, 1, 3) |

The lakh unit is inferred from the values (for example, a 2014 Ritz sells for
3.35 against a present price of 5.59). It is not stated in the file.

## Problems found
1. **101 of 301 rows are two-wheelers**, not cars. These are motorcycles and
   scooters such as Royal Enfield, Bajaj Pulsar and Honda Activa. All of them
   are sold by individuals, which creates a misleading link between
   `Seller_Type` and price. They also hold the most extreme mileage values
   (500,000 km on a 2008 Activa scooter, which is almost certainly an entry
   error).
2. **2 exact duplicate rows** (an ertiga and a fortuner).
3. **No impossible values** remain after these steps. There are no negative
   prices or km, no future years, and no car sells above its present price.

## Clean dataset (198 cars)
- **Fuel:** Petrol 138, Diesel 58, CNG 2
- **Transmission:** Manual 168, Automatic 30
- **Seller:** Dealer 193, Individual 5
- **Previous owners:** 0 → 193, 1 → 4, 3 → 1
- **Kms_Driven:** 2,071 to 197,176
- **Selling_Price:** 0.35 to 35 lakhs (median 5.25)

## Suitability
The dataset is **suitable for a teaching-scale Multiple Linear Regression
project**. It has a clear numeric target, a mix of numeric and categorical
features, no missing data, and features that would be known when a car is
put up for sale.

## Limitations
- **Small sample.** An 80/20 split leaves about 40 test cars, so a single test
  score can swing noticeably. Cross-validation on the training set will be
  reported alongside it.
- **Narrow coverage.** 98 model names, mostly mass-market Maruti, Honda,
  Hyundai and Toyota. Predictions for luxury, very old or rare models are
  unreliable.
- **Nearly constant features.** Seller type, owner count and CNG each have five
  or fewer minority examples, so their effects cannot be estimated reliably.
- **Undocumented collection.** Date, region and listing-vs-sold status are
  unknown. Prices reflect listings from around 2018–2020, not today's market.
- **`Present_Price` dependence.** It is the strongest predictor (correlation
  0.82 with the target for cars). A user must look it up to get a prediction.
