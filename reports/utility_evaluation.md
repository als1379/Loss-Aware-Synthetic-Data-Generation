# Utility Evaluation

## Metrics

Utility is evaluated with three metric families:

- Distributional similarity:
  - KS statistic for numeric columns
  - total variation distance and chi-square p-value for categorical columns
- Correlation preservation:
  - Frobenius norm between one-hot mixed-association matrices
- Downstream performance:
  - Train on Synthetic, Test on Real (TSTR)
  - Compared against the Train-Real-Test-Real baseline

Lower values are better for KS, total variation distance, correlation Frobenius distance, and TSTR gaps.

## Current Local-Run Summary

The current local synthetic generation setup used short CTGAN/TVAE training (`10` epochs), so neural synthesizer quality should be interpreted as a quick smoke-test rather than final evidence.

| Generator | Mean Numeric KS | Mean Categorical TVD | Correlation Distance | Mean F1 Gap | Mean AUC Gap |
|---|---:|---:|---:|---:|---:|
| GaussianCopula | 0.165469 | 0.016826 | 11.000829 | 0.235528 | 0.112930 |
| CTGAN | 0.167344 | 0.111472 | 10.323508 | 0.319399 | 0.363126 |
| TVAE | 0.269875 | 0.472229 | 28.368408 | 0.547072 | 0.169777 |

In this quick local run, GaussianCopula has the strongest downstream utility and categorical distribution match. CTGAN has slightly better correlation distance, but worse downstream gaps. TVAE needs longer training before it is meaningful.

## Outputs

- `results/folktables_acs_income/distribution_metrics.csv`
- `results/folktables_acs_income/tstr_metrics.csv`
- `results/folktables_acs_income/utility_summary.csv`
- `results/folktables_acs_income/plots/distributions/`
