# Synthetic Data Generation

## Inputs

Synthetic generators are trained on the real training split:

`data/processed/folktables_acs_income/train.csv`

The current local-development configuration uses:

- Fit rows: `10000`
- Synthetic sample size per generator: `16000`
- Random seed: `42`

These values are intentionally small enough to run locally. For a full Colab/GPU run, increase `synthetic_generation.fit_max_rows`, increase `synthetic_generation.sample_size`, and increase CTGAN/TVAE epochs.

## Generators

Implemented SDV single-table generators:

- `gaussian_copula`
- `ctgan`
- `tvae`

Each generator writes:

- a synthetic CSV under `data/synthetic/folktables_acs_income/`
- a generator config JSON under the same directory
- shared SDV metadata in `sdv_metadata.json`

## Current Output Files

- `data/synthetic/folktables_acs_income/gaussian_copula.csv`
- `data/synthetic/folktables_acs_income/ctgan.csv`
- `data/synthetic/folktables_acs_income/tvae.csv`

These datasets include the target column so utility evaluation can run Train-Synthetic-Test-Real evaluation.
