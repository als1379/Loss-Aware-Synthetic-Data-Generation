# Pipeline And Reporting

## Reproducible Pipeline

The full pipeline entry point is:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_pipeline --config config/default.json
```

For faster local iteration when phase outputs already exist:

```powershell
$env:PYTHONPATH = "src"
python -m lossaware.run_pipeline --config config/default.json --reuse-existing
```

The full command runs:

- baseline training
- synthetic generation
- utility evaluation
- privacy evaluation
- loss-aware selection
- final summary report

## Final Outputs

- `reports/final_summary.md`
- `results/folktables_acs_income/final_results_summary.csv`
- `results/folktables_acs_income/loss_aware_selected.csv`

## Ethics Framing

The report emphasizes that synthetic data should not be treated as automatically private or automatically useful. The framework measures both sides, makes the trade-off explicit, and selects a synthetic candidate only if it passes privacy constraints.
