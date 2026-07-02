# Loss-Aware Synthetic Data Generation Summary

## Dataset

- Benchmark: `folktables_acs_income`
- Adult remains available as a simple baseline loader, but ACSIncome is the selected harder benchmark.

## Real-Data Baseline

The reference task is Train-Real-Test-Real classification.

- Best baseline model: `logistic_regression`
- Accuracy: 0.8142
- F1: 0.7749
- ROC AUC: 0.8941

## Synthetic Utility

Utility was measured with distribution mismatch, correlation distortion, and Train-Synthetic-Test-Real performance gaps.

- Selected generator: `gaussian_copula`
- Mean numeric KS: 0.1487
- Mean categorical TVD: 0.0118
- Correlation distance: 8.0889
- Mean TSTR F1 gap: 0.2160
- Mean TSTR AUC gap: 0.1224

## Privacy Indicators

Privacy was measured with distance-to-closest-record and a simple membership-risk proxy.

- DCR median: 2.0906
- Exact match rate: 0.0000
- Membership advantage: 0.0000

## Loss-Aware Strategy

The loss-aware selector normalized each utility metric, applied the configured composite-loss weights, and selected the lowest-loss candidate satisfying privacy constraints.

- Selected candidate: `gaussian_copula`
- Composite loss: 0.4024
- TVAE was rejected by the privacy constraints in the current local run.

## Ethics Connection

This framework makes the privacy-utility trade-off explicit instead of treating synthetic data as automatically safe. Utility metrics show whether synthetic data remains useful for downstream ML, while privacy indicators check whether generation may be too close to real individuals. The loss-aware stage turns those measurements into an auditable decision rule: choose a useful synthetic dataset only when it also satisfies privacy constraints. That supports responsible data sharing, privacy-preserving AI, and more trustworthy ML experimentation.

## Key Output Files

- `results/folktables_acs_income/final_results_summary.csv`
- `results/folktables_acs_income/loss_aware_ranking.csv`
- `results/folktables_acs_income/loss_aware_selected.csv`
- `results/folktables_acs_income/plots/distributions/`
