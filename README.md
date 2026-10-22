# Used Car Price Prediction

An end-to-end machine learning project for estimating used-car resale prices in lakh.

## Current Status

Implemented phases:

- Dataset auditing and profiling
- Data cleaning and leakage review
- Exploratory data analysis
- Leakage-safe feature engineering
- Train/test preprocessing pipeline
- Baseline model comparison
- Hyperparameter tuning experiment
- Final evaluation and permutation importance
- Streamlit prediction application

## Dataset

The supplied dataset contains 100,000 rows and 22 columns. The prediction target is
`Selling_Price_Lakh`. The raw file is preserved at
`dataset/used_car_dataset_100000.csv`.

The modeling dataset excludes `Car_ID`, `Demand_Score`, and `Depreciation_Percent`
because they are identifiers or potential target-derived leakage fields. Missing
`Service_History` values are represented as `Unknown`.

## Results

The selected model is HistGradientBoostingRegressor.

| Metric | Test result |
| --- | ---: |
| MAE | 0.240 lakh |
| RMSE | 0.378 lakh |
| R2 | 0.9928 |
| MAPE | 4.84% |

These results are specific to the supplied dataset and test split. They should not
be interpreted as guaranteed real-world transaction-price accuracy.

## Project Structure

```text
dataset/                    Raw CSV data
data/processed/             Cleaned and feature-enhanced datasets
models/                     Saved preprocessing and model bundles
reports/                    Audit, EDA, evaluation, and explainability outputs
src/data_audit.py           Dataset validation
src/data_understanding.py   Dataset profiling
src/data_cleaning.py        Cleaning and leakage removal
src/eda.py                  Exploratory analysis plots and findings
src/feature_engineering.py  Deterministic feature creation
src/preprocessing.py        Train-only preprocessing pipeline
src/train_baselines.py      Baseline model comparison
src/tune_models.py          Cross-validated tuning experiment
src/evaluate_model.py       Metrics, residuals, and feature importance
src/predict.py              Reusable prediction API
app/app.py                  Streamlit interface
```

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce the Workflow

Run these commands from the project root:

```powershell
venv\Scripts\python.exe src\data_audit.py
venv\Scripts\python.exe src\data_understanding.py
venv\Scripts\python.exe src\data_cleaning.py
venv\Scripts\python.exe src\eda.py
venv\Scripts\python.exe src\feature_engineering.py
venv\Scripts\python.exe src\preprocessing.py
venv\Scripts\python.exe src\train_baselines.py
venv\Scripts\python.exe src\tune_models.py
venv\Scripts\python.exe src\evaluate_model.py
```

## Run the Application

```powershell
venv\Scripts\streamlit.exe run app\app.py
```

The application accepts vehicle details and returns an estimated price plus an
empirical price range based on held-out test residuals.

## Limitations

The dataset provenance and market coverage must be documented before making real
commercial claims. Used-car prices vary by region, listing date, negotiation,
vehicle condition, service history, and supply and demand. The model provides an
estimate, not a guaranteed sale price.

## Ideas for Next Iteration

### Data and modelling

- Add verified listing-date and transaction-date data for time-based validation.
- Compare random splitting with a chronological holdout to measure market drift.
- Add vehicle variant, BHP, torque, seat count, registration state, and accident cost.
- Build depreciation curves by brand, fuel type, vehicle segment, and location.
- Test CatBoost or XGBoost with a controlled runtime budget.
- Add prediction intervals using quantile regression or conformal prediction.
- Monitor feature drift and prediction error as new listings arrive.

### Product features

- Add a comparison view for multiple cars.
- Show a recommended listing price and an estimated negotiation range.
- Add a resale-value score showing how well a vehicle retains value.
- Provide a breakdown of the main factors that increased or reduced the estimate.
- Add downloadable prediction reports for sellers and dealerships.
- Add an API endpoint for integration with a car marketplace.

### Advanced research directions

- Use SHAP explanations for individual predictions.
- Incorporate geospatial features for city-level price differences.
- Analyse free-text vehicle descriptions with natural language processing.
- Use vehicle photographs to estimate visible condition and damage.
- Develop a time-series component for changing market prices.
- Compare the model against a simple dealer or depreciation-rule baseline.
