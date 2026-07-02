"""Evaluate privacy-risk indicators for synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lossaware.config import load_config, set_random_seed
from lossaware.datasets import get_loader_from_config
from lossaware.privacy import evaluate_privacy, save_privacy_results


def run_privacy_evaluation(config_path: str | Path = "config/default.json") -> pd.DataFrame:
    """Evaluate privacy indicators and join them with utility metrics."""
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    set_random_seed(seed)

    dataset_name = config["dataset"]["name"]
    target_column = config["dataset"]["target_column"]
    loader = get_loader_from_config(config)
    bundle = loader.load(config["paths"]["raw_data_dir"])

    processed_dir = Path(config["paths"]["processed_data_dir"]) / dataset_name
    synthetic_dir = Path(config["paths"]["synthetic_data_dir"]) / dataset_name
    results_dir = Path(config["paths"]["results_dir"]) / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    privacy_config = config["privacy_evaluation"]

    summaries = []
    for synthetic_path in sorted(synthetic_dir.glob("*.csv")):
        synthetic = pd.read_csv(synthetic_path)
        summary = evaluate_privacy(
            generator=synthetic_path.stem,
            real_train=train,
            real_test=test,
            synthetic=synthetic,
            categorical_columns=bundle.categorical_columns,
            numerical_columns=bundle.numerical_columns,
            target_column=target_column,
            max_real_rows=privacy_config.get("max_real_rows"),
            max_synthetic_rows=privacy_config.get("max_synthetic_rows"),
            random_seed=seed,
            exact_match_tolerance=privacy_config.get("exact_match_tolerance", 1e-9),
        )
        summaries.append(summary.__dict__)

    privacy_results = pd.DataFrame(summaries)
    save_privacy_results(privacy_results, results_dir / "privacy_metrics.csv")

    utility_path = results_dir / "utility_summary.csv"
    if utility_path.exists():
        utility = pd.read_csv(utility_path)
        tradeoff = utility.merge(privacy_results, on="generator", how="inner")
        tradeoff.to_csv(results_dir / "privacy_utility_tradeoff.csv", index=False)

    return privacy_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    results = run_privacy_evaluation(args.config)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
