# Privacy Evaluation

## Metrics

Privacy is evaluated with simple distance-based indicators:

- DCR: distance from each synthetic record to its closest real training record.
- Exact match rate: fraction of synthetic records with near-zero distance to a real training record.
- Train-closer rate: fraction of synthetic records closer to the real training set than to the held-out real test set.
- Membership advantage: `max(0, train_closer_rate - 0.5)`.

Lower exact match rate and membership advantage are better. Higher DCR generally indicates less direct memorization risk, though it must be interpreted alongside utility.

## Current Local-Run Summary

| Generator | DCR Min | DCR P01 | DCR Median | Exact Match Rate | Train-Closer Rate | Membership Advantage |
|---|---:|---:|---:|---:|---:|---:|
| GaussianCopula | 0.076957 | 1.416306 | 2.147214 | 0.000000 | 0.653375 | 0.153375 |
| CTGAN | 0.102441 | 1.415829 | 2.308060 | 0.000000 | 0.619875 | 0.119875 |
| TVAE | 0.000000 | 0.202843 | 1.420664 | 0.001375 | 0.811250 | 0.311250 |

In this quick local run, TVAE shows the highest privacy risk: it has lower nearest-neighbour distances, a non-zero exact match rate, and the largest membership advantage. CTGAN has the lowest membership advantage, while GaussianCopula has stronger utility.

## Privacy-Utility Trade-Off

The framework writes a joined table combining utility and privacy metrics:

`results/folktables_acs_income/privacy_utility_tradeoff.csv`

This table is the basis for loss-aware selection.
