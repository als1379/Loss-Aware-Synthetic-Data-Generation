# Preprocessing And Baseline Notes

## Dataset

The configured harder benchmark is `folktables_acs_income`, using ACSIncome via Folktables with:

- Survey year: 2018
- Horizon: 1-Year
- Survey: person
- State: CA
- Target: `income_gt_50k`

For local development, `dataset.max_rows` is set to `20000`, using a reproducible stratified sample before train/test splitting. Set this value to `null` for a full-size run, preferably in Colab.

## Cleaning

- Missing markers such as `?` are normalized before modeling.
- ACS integer-coded categorical variables are treated as categorical strings.
- Numeric variables are coerced to numeric values.
- Categorical missing values are imputed with the most frequent value.
- Numeric missing values are imputed with the median.

## Encoding And Scaling

- Categorical features use one-hot encoding with `handle_unknown="ignore"`.
- Numeric features use standard scaling after median imputation.
- The target is label-encoded for sklearn classifiers.

## Baseline Task

The downstream task is binary classification:

`income_gt_50k = 1` means personal income is greater than $50K.

The real-data reference uses Train-Real-Test-Real evaluation with:

- Logistic Regression
- Random Forest
- Gradient Boosting
