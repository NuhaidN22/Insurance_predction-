# Medical Insurance Cost Prediction

Predicting a person's yearly medical insurance charges from basic personal details, using scikit-learn pipelines, cross-validation and XGBoost.

## Problem

Given a person's age, sex, BMI, number of children, smoking status and region, predict their yearly medical insurance charges (in US dollars). This is a regression problem.

## Data

- **Source:** Kaggle, "Medical Cost Personal Datasets" (`insurance.csv`)
- **Size:** 1,338 rows, 6 features + 1 target
- **Missing values:** none

| Column | Type | Description |
| --- | --- | --- |
| `age` | numeric | Age in years |
| `sex` | categorical | male / female |
| `bmi` | numeric | Body mass index |
| `children` | numeric | Number of children covered |
| `smoker` | categorical | yes / no |
| `region` | categorical | One of 4 US regions |
| `charges` | numeric | **Target:** yearly medical cost ($) |

**Leakage check:** all six features are known before the charge is set, so there is no target leakage.

## Approach

1. **Explore:** shape, data types, missing values, cardinality of text columns (2, 2 and 4 values, so one-hot encoding is cheap).
2. **Split columns** into numeric and categorical lists using `pd.api.types.is_numeric_dtype` / `is_string_dtype` (works on old and new pandas versions).
3. **Preprocessing with a `ColumnTransformer`:**
   - numeric: `SimpleImputer(strategy='constant')`
   - categorical: `SimpleImputer(strategy='most_frequent')` then `OneHotEncoder(handle_unknown='ignore')`
4. **Pipeline:** preprocessing + model bundled together, so preprocessing is re-fitted inside every fold (no train-test contamination).
5. **Evaluation:** 5-fold cross-validation (MAE), chosen because the dataset is small and a single split is noisy. A 75/25 train/test split was kept for a final check.
6. **Models compared:** decision tree, random forest, XGBoost (untuned and tuned).

## Results

5-fold cross-validated Mean Absolute Error (lower is better):

| Model | Settings | CV MAE |
| --- | --- | --- |
| Decision tree | default | $3,080 |
| Random forest | 100 trees | $2,654 |
| XGBoost (untuned) | 1000 trees, learning rate 0.05 | $2,978 |
| **XGBoost (tuned)** | **100 trees, learning rate 0.05** | **$2,547** |

- Random forest on the single train/test split: MAE $2,682 (close to its CV score, so the split was representative).
- Tuned XGBoost on the held-out test set: MAE $_____ <!-- fill in -->

## Key findings

- **Random forest beat a single decision tree by about $430**: averaging many trees reduces overfitting.
- **Untuned XGBoost lost to the random forest.** With 1000 trees and no early stopping it kept fitting noise in this small dataset. Cutting it to 100 trees improved MAE from $2,978 to $2,547, making it the best model.
- **Most important features:** _____ <!-- fill in from feature_importances_ -->

## What I learned

- Building preprocessing and models as one `Pipeline` keeps validation data out of every `fit`.
- Cross-validation gives more trustworthy comparisons than one split on small data.
- A stronger algorithm only wins after tuning: "usually best" is not "always best".
- pandas 3 stores text columns as `str` instead of `object`, so type checks should use `pd.api.types` helpers.

## How to run

```bash
pip install pandas scikit-learn xgboost
jupyter notebook insurance_cost_prediction.ipynb
```

Place `insurance.csv` in the same folder as the notebook (or update the path in the first cell), then run all cells.
