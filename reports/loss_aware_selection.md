# Loss-Aware Selection

## Strategy

The loss-aware strategy ranks candidate synthetic datasets with:

`loss = w1 * distribution_mismatch + w2 * correlation_distortion + w3 * downstream_degradation`

Current weights:

- `distribution_mismatch`: 0.4
- `correlation_distortion`: 0.3
- `downstream_degradation`: 0.3

Each raw metric is min-max normalized across candidates before weighting, so metrics on different scales can be combined.

## Privacy Constraints

Candidates must satisfy:

- `dcr_median >= 1.5`
- `exact_match_rate <= 0.0`
- `membership_advantage <= 0.2`

Candidates that fail privacy constraints are not eligible for selection even if their utility loss is low.

## Current Selection

Selected generator:

`gaussian_copula`

Why:

- It passed the privacy constraints.
- It had the strongest distribution match.
- It had the smallest downstream TSTR degradation.
- CTGAN also passed the privacy constraints, but its downstream utility loss was higher.
- TVAE failed the privacy constraints because of exact matches and high membership advantage.

## Outputs

- `results/folktables_acs_income/loss_aware_ranking.csv`
- `results/folktables_acs_income/loss_aware_selection.json`
- `results/folktables_acs_income/loss_aware_selected.csv`
