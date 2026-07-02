"""Run preprocessing and real-data baseline classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

from lossaware.baselines import evaluate_baselines, save_baseline_results
from lossaware.config import load_config, set_random_seed
from lossaware.datasets import get_loader_from_config
from lossaware.preprocessing import save_split, split_dataset


def run_baseline_training(config_path: str | Path = "config/default.json"):
    """Load data, save cleaned splits, and evaluate real-data baselines."""
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    set_random_seed(seed)

    loader = get_loader_from_config(config)
    bundle = loader.load(config["paths"]["raw_data_dir"])
    split = split_dataset(
        bundle=bundle,
        test_size=config["dataset"]["test_size"],
        random_seed=seed,
        max_rows=config["dataset"].get("max_rows"),
    )

    dataset_dir = Path(config["paths"]["processed_data_dir"]) / bundle.name
    save_split(split, dataset_dir)

    results = evaluate_baselines(
        dataset_name=bundle.name,
        split=split,
        model_names=config["models"]["baseline_classifiers"],
        random_seed=seed,
    )
    output_path = Path(config["paths"]["results_dir"]) / bundle.name / "baseline_results.csv"
    save_baseline_results(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    results = run_baseline_training(args.config)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
