# Viva / Interview Preparation

Short answers in plain language, tied to what *this* project actually did.
Numbers refer to the project outputs.

## The project in 30 seconds
I predicted used-car resale prices in India from the CarDekho dataset.
- **Data:** I found 101 motorcycles mixed into the "car" data and removed
  them, along with 2 duplicates, leaving 198 cars.
- **Method:** I split the data 80/20 before any analysis. I then chose
  between three versions of Multiple Linear Regression using
  cross-validation on the training data only. The best one predicts log
  price from age, km, log showroom price, fuel and transmission.
- **Results:** on 40 unseen cars, the average error is about ₹94,000 and
  R² is 0.915.
- **Weaknesses:** it under-predicts expensive models, and it relies
  heavily on the showroom price.
- **Delivery:** a Streamlit app that validates inputs and warns when a
  car is outside the training data.

---

## Regression basics

**Q: What is Multiple Linear Regression?**
It predicts a number as a weighted sum of several inputs plus a constant:
ŷ = b₀ + b₁x₁ + … + bₙxₙ. The weights are chosen by *least squares*: the
values that make the sum of squared errors on the training data as small
as possible.

**Q: Your model predicts log(price). Is it still linear regression?**
Yes. "Linear" means linear *in the coefficients*. I transformed the target
and one input (log), but the model is still a weighted sum fitted by
ordinary least squares. It is plain `LinearRegression` from scikit-learn.

**Q: Why log price?**
- The price is right-skewed (skewness 2.38 vs −0.19 after log).
- EDA showed cars lose a roughly constant *percentage* of value per year,
  and a log model expresses effects as percentages.
- Most importantly, I didn't just assume it. I compared raw, log-target
  and log-log versions with cross-validation on the training set. The
  log-log version had CV RMSE 0.96 lakh, against 1.82 for raw.

**Q: What does the intercept mean?**
It is the predicted log price when every input is zero: age 0, 0 km, a
log present price of 0 (₹1 lakh), petrol, manual. No real car looks like
that, so it has no useful meaning on its own. It just positions the
regression surface.

**Q: Interpret your coefficients.**
Each effect holds all other inputs fixed:
- **Age:** −9.7 % price per year (e^−0.1016 − 1).
- **Km:** −2.8 % per 10,000 km.
- **Present price:** 1 % higher showroom price → 0.82 % higher resale price
  (an elasticity).
- **Diesel:** +21 % vs petrol.
- **Automatic:** +5.6 % vs manual.

**Q: Does diesel *cause* a 21 % higher price?**
No. It is an association in this sample, after adjusting for the other
inputs. It may reflect which diesel models appear in the data, such as
Innovas and Fortuners. Regression on observational data does not prove
causation.

**Q: Assumptions of linear regression?**
- A linear relationship
- Independent errors
- Constant error variance
- Roughly normal errors (needed for inference, not for prediction)
- No perfect multicollinearity

The log scale helped with linearity and constant variance. The residual
plots still show a pattern: expensive cars are under-predicted.

## Data and features

**Q: What data problems did you find?**
- **101 two-wheelers in a car dataset.** All were sold by individual
  sellers, which made "seller type" look strongly related to price
  (−0.55 correlation) when it barely was for cars (−0.03).
- **2 duplicate rows.**

**Q: Why use car age instead of year?**
Age is easier to interpret. In a linear model the two are equivalent
(age = 2020 − year), so the reference year only shifts the intercept. I
never use both, because they are perfectly collinear.

**Q: Why did you drop Seller_Type and Owner?**
- In training, only 5 cars had an individual seller and only 5 had any
  previous owner. Owner = 3 was a single car.
- The test set had none of them, so their coefficients could never be
  checked.
- I decided this on domain grounds *before* looking at test results, and
  recorded it in the decision log.

**Q: Isn't Present_Price "cheating"?**
No. It is the current showroom price of the *model*, which a seller can
look up before listing. It does not come from the selling price. My
secondary experiment shows how much it matters: without it, test R²
drops from 0.915 to 0.111. The trade-off is that the user has to look it
up.

**Q: What is one-hot encoding, and why drop a column?**
- It turns a category into 0/1 columns: Diesel = 1 if the car is diesel.
- Keeping all levels plus an intercept creates perfect multicollinearity
  (the "dummy-variable trap"), so one *reference* level is dropped.
- I chose Petrol and Manual as references, since they are the most
  common. Each category coefficient then means "compared with a petrol
  manual car". With CNG (2 cars) as the reference, the fuel columns had a
  variance inflation factor (VIF) of about 17.

**Q: Did you scale features?**
No. Ordinary least squares gives the same predictions with or without
scaling. Unscaled inputs keep the coefficients in real units (per year,
per km).

**Q: What is multicollinearity and did you have it?**
It means inputs are strongly related to each other, which makes their
individual coefficients unstable. Age and km have a correlation of 0.71,
but their VIFs were 2.3–2.8, which is acceptable. The km coefficient is
therefore the *extra* effect of km at the same age.

## Train/test split, overfitting, leakage

**Q: Why split the data, and why 80/20?**
- To measure performance on cars the model has never seen.
- 80/20 is a common default. With 198 cars it leaves 158 to train on and
  40 to test.
- The fixed seed (42) makes the split reproducible.

**Q: What is data leakage, and how did you prevent it?**
Leakage is when information from the test set, or from the future, gets
into training, which makes results look better than they really are. To
prevent it:
1. I split *before* EDA.
2. The encoders are fitted inside the pipeline on training data only.
3. The model variant was chosen by cross-validation on training data,
   using a rule I wrote down in advance.
4. The test set was evaluated once.
5. The target is never an input.

**Q: What is cross-validation and why use it?**
- The training set is split into 5 parts. The model trains on 4 and is
  scored on the 5th, rotating through all 5 parts. I repeated this 10
  times with different shuffles.
- It gives a more stable estimate than a single split, and it lets me
  compare model choices *without* touching the test set.

**Q: Is your model overfitting?**
No sign of it. Training RMSE is 0.91, cross-validated RMSE is 0.96, and
test RMSE on the 39 in-range cars is 0.90. With only 7 coefficients for
158 cars, overfitting is unlikely.

## Metrics

**Q: Explain MAE, RMSE and R² with your numbers.**
- **MAE 0.935 lakh:** the average size of an error, about ₹93,500.
- **RMSE 1.696 lakh:** squares errors before averaging, so big misses
  count more. It is much larger than MAE because of one car: the land
  cruiser was off by 9.1 lakh. Without it, RMSE is 0.90.
- **R² 0.915:** the model explains about 92 % of the variation in test
  prices around their average.

**Q: R² is 0.915. Is the model good?**
For this data it is reasonable, but R² alone is not enough:
- A lot of it comes from the showroom price separating cheap cars from
  expensive ones.
- A median 15 % error can still be ₹1 lakh or more on a single car.
- The model consistently under-predicts expensive cars.

I judge the model on MAE, the residual plots and the error table, not
only on R².

**Q: Why did you report two test results?**
- Before training any model, I noticed that one test car (a land cruiser
  with a showroom price of ₹92.6 lakh) was 2.5× above the most expensive
  car in training. Any linear model must extrapolate there.
- I fixed the reporting rule in advance: the headline uses all 40 cars,
  and a second line shows the 39 in-range cars.
- Fixing the rule beforehand means I couldn't pick whichever version
  looked better.

## Limitations and improvements

**Q: Main limitations?**
- A small sample (198 cars).
- Expensive popular models are under-predicted.
- No brand, condition or location information.
- Old prices (2018–2020).
- Rare categories (CNG has 2 cars).
- The user must know the showroom price.

**Q: How would you improve it while keeping linear regression?**
- A larger and more recent dataset.
- A brand or segment feature.
- Prediction intervals.
- Ridge regression as a comparison.

**Q: What happens if a user enters something strange in the app?**
- **Impossible values are blocked with a message:** a negative price, a
  year after 2020, or an unknown fuel type.
- **Unusual values are allowed but flagged:** valid but outside the
  training range (for example a ₹90 lakh car). The app shows a warning
  that the model is extrapolating.
- **CNG** triggers a small-sample warning.
- **An estimate above the showroom price** triggers a plausibility
  warning.
