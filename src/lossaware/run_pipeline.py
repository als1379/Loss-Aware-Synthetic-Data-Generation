"""Run the full reproducible synthetic-data evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from lossaware.config import load_config
from lossaware.reporting import build_final_summary
from lossaware.run_baseline_training import run_baseline_training
from lossaware.run_loss_aware_selection import run_loss_aware_selection
from lossaware.run_privacy_evaluation import run_privacy_evaluation
from lossaware.run_synthetic_generation import run_synthetic_generation
from lossaware.run_utility_evaluation import run_utility_evaluation


def _has_required_outputs(config: dict, step: str) -> bool:
    dataset_name = config["dataset"]["name"]
    results_dir = Path(config["paths"]["results_dir"]) / dataset_name
    processed_dir = Path(config["paths"]["processed_data_dir"]) / dataset_name
    synthetic_dir = Path(config["paths"]["synthetic_data_dir"]) / dataset_name

    required = {
        "baseline_training": [
            processed_dir / "train.csv",
            processed_dir / "test.csv",
            results_dir / "baseline_results.csv",
        ],
        "synthetic_generation": [
            synthetic_dir / "gaussian_copula.csv",
            synthetic_dir / "ctgan.csv",
            synthetic_dir / "tvae.csv",
        ],
        "utility_evaluation": [results_dir / "utility_summary.csv"],
        "privacy_evaluation": [results_dir / "privacy_utility_tradeoff.csv"],
        "loss_aware_selection": [results_dir / "loss_aware_ranking.csv"],
    }
    return all(path.exists() for path in required[step])


def run_pipeline(
    config_path: str | Path = "config/default.json",
    reuse_existing: bool = False,
) -> Path:
    """Run the full workflow and return the final report path."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)

    workflow = [
        ("baseline_training", run_baseline_training),
        ("synthetic_generation", run_synthetic_generation),
        ("utility_evaluation", run_utility_evaluation),
        ("privacy_evaluation", run_privacy_evaluation),
        ("loss_aware_selection", run_loss_aware_selection),
    ]
    for step_name, runner in workflow:
        if reuse_existing and _has_required_outputs(config, step_name):
            print(f"{step_name}: reusing existing outputs")
            continue
        print(f"{step_name}: running")
        runner(config_path)

    print("reporting: writing final summary")
    return build_final_summary(
        results_dir=config["paths"]["results_dir"],
        reports_dir=config["paths"]["reports_dir"],
        dataset_name=config["dataset"]["name"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing phase outputs when present.",
    )
    args = parser.parse_args()

    report_path = run_pipeline(args.config, reuse_existing=args.reuse_existing)
    print(f"final_report={report_path}")


if __name__ == "__main__":
    main()
