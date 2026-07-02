# Loss-Aware Synthetic Data Generation

Experimental framework for evaluating synthetic tabular data through a privacy-utility trade-off lens.

The project compares real and synthetic tabular datasets using:

- Utility loss: feature-level distributional similarity, mixed-type association preservation, and downstream model performance.
- Privacy risk: distance-to-closest-record indicators and simple inference-risk proxies.
- Loss-aware generation: multiple synthetic candidates are generated and selected by a composite utility loss, subject to a privacy threshold.

This is designed for a Trustworthy AI course project. Adult/Census Income is included as a simple baseline, while ACSIncome via Folktables is the selected harder benchmark.

## Repository Structure

```text
data/
  raw/          # downloaded or manually placed source datasets
  processed/    # cleaned train/test artifacts
  synthetic/    # generated synthetic datasets
src/
  lossaware/    # framework code
notebooks/      # exploratory analysis
results/        # metrics, plots, tables
reports/        # final writeups and figures
tests/          # lightweight regression tests
config/         # reproducible experiment configuration
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The main configuration lives in `config/default.json`. Randomness should be controlled through the `project.random_seed` value.

## Reproducibility

- Use the fixed seed in `config/default.json` for data splits, model training, and synthetic-data generation.
- Keep downloaded raw datasets in `data/raw/`.
- Write generated datasets to `data/synthetic/`.
- Write metric tables and plots to `results/`.

## Baseline Training

After installing the dependencies, run the real-data preprocessing and baseline classifiers with:

```bash
set PYTHONPATH=src
python -m lossaware.run_baseline_training --config config/default.json
```

On PowerShell, use:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_baseline_training --config config/default.json
```

This downloads the configured dataset if needed, writes cleaned train/test splits under `data/processed/`, and writes baseline metrics under `results/`.

## Synthetic Generation

Generate synthetic datasets from the real training split with:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_synthetic_generation --config config/default.json
```

The current local configuration trains GaussianCopula, CTGAN, and TVAE on a reproducible subset of the real training data and writes synthetic CSV files to `data/synthetic/`.

## Utility Evaluation

Evaluate distribution similarity, correlation preservation, and Train-Synthetic-Test-Real utility with:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_utility_evaluation --config config/default.json
```

Outputs are written under `results/<dataset>/`, including metric CSVs and distribution plots.

## Privacy Evaluation

Evaluate distance-based privacy indicators with:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_privacy_evaluation --config config/default.json
```

This writes privacy metrics and a joined privacy-utility trade-off table under `results/<dataset>/`.

## Loss-Aware Selection

Select the best synthetic candidate by weighted utility loss subject to privacy constraints with:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_loss_aware_selection --config config/default.json
```

This writes a ranked candidate table and copies the selected synthetic dataset to `results/<dataset>/loss_aware_selected.csv`.

## Full Pipeline

Run the full reproducible pipeline with:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_pipeline --config config/default.json
```

For local iteration using existing outputs:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_pipeline --config config/default.json --reuse-existing
```

The final report is written to `reports/final_summary.md`, and a compact result table is written to `results/<dataset>/final_results_summary.csv`.

## Stronger Colab Run

The default config is intentionally small enough for local development. For stronger results, use `config/strong.json` in Colab or another GPU environment:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_pipeline --config config/strong.json
```

The strong config removes the ACS row cap, fits synthetic generators on more rows, generates larger synthetic datasets, and trains CTGAN/TVAE for more epochs.

## Datasets

Implemented loaders:

- `adult`: UCI Adult/Census Income, used as the simple baseline.
- `folktables_acs_income`: ACSIncome via Folktables, selected as the harder benchmark because it is larger, more realistic, configurable by state/year, and ethically relevant for socioeconomic prediction.

The default configuration uses `folktables_acs_income` with 2018 1-Year ACS person records for California. This keeps early experiments manageable while still moving beyond the Adult toy baseline. The Folktables settings can be changed in `config/default.json`.

For local development, `config/default.json` caps baseline training to a reproducible stratified sample of `20000` rows. Use `config/strong.json` for a full-size Colab run.

Alternative candidates kept in the dataset registry for later extension:

- `uci_bank_marketing`
- `uci_default_credit_card`

## Current Status

The framework includes dataset loading, preprocessing, baseline training, synthetic generation, utility evaluation, privacy evaluation, loss-aware selection, and final reporting.
