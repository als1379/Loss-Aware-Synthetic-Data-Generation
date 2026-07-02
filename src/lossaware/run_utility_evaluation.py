"""Evaluate synthetic-data utility."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lossaware.config import load_config, set_random_seed
from lossaware.datasets import get_loader_from_config
from lossaware.preprocessing import SplitData
from lossaware.utility import (
    correlation_distance,
    distribution_metrics,
    evaluate_tstr,
    plot_feature_distributions,
    summarize_utility,
)


def load_real_split(processed_dir: Path, target_column: str, categorical: list[str], numerical: list[str]) -> SplitData:
    """Load the real-data train/test split from disk."""
    train = pd.read_csv(processed_dir / "train.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    return SplitData(
        X_train=train.drop(columns=[target_column]),
        X_test=test.drop(columns=[target_column]),
        y_train=train[target_column],
        y_test=test[target_column],
        target_column=target_column,
        categorical_columns=categorical,
        numerical_columns=numerical,
    )


def run_utility_evaluation(config_path: str | Path = "config/default.json") -> pd.DataFrame:
    """Evaluate utility of every generated synthetic CSV."""
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
    plots_dir = results_dir / "plots" / "distributions"
    results_dir.mkdir(parents=True, exist_ok=True)

    real_split = load_real_split(
        processed_dir,
        target_column=target_column,
        categorical=bundle.categorical_columns,
        numerical=bundle.numerical_columns,
    )
    real_train = pd.read_csv(processed_dir / "train.csv")
    baseline_results = pd.read_csv(results_dir / "baseline_results.csv")

    distribution_tables: list[pd.DataFrame] = []
    tstr_tables: list[pd.DataFrame] = []
    summaries = []

    for synthetic_path in sorted(synthetic_dir.glob("*.csv")):
        generator = synthetic_path.stem
        synthetic = pd.read_csv(synthetic_path)
        distribution = distribution_metrics(
            real=real_train,
            synthetic=synthetic,
            generator=generator,
            categorical_columns=bundle.categorical_columns,
            numerical_columns=bundle.numerical_columns,
            target_column=target_column,
        )
        corr_distance = correlation_distance(
            real=real_train,
            synthetic=synthetic,
            categorical_columns=bundle.categorical_columns,
            numerical_columns=bundle.numerical_columns,
            target_column=target_column,
        )
        tstr = evaluate_tstr(
            generator=generator,
            synthetic_train=synthetic,
            real_split=real_split,
            model_names=config["models"]["baseline_classifiers"],
            baseline_results=baseline_results,
            random_seed=seed,
        )
        plot_feature_distributions(
            real=real_train,
            synthetic=synthetic,
            generator=generator,
            categorical_columns=bundle.categorical_columns,
            numerical_columns=bundle.numerical_columns,
            target_column=target_column,
            output_dir=plots_dir,
        )

        distribution_tables.append(distribution)
        tstr_tables.append(tstr)
        summaries.append(summarize_utility(generator, distribution, corr_distance, tstr))

    distribution_results = pd.concat(distribution_tables, ignore_index=True)
    tstr_results = pd.concat(tstr_tables, ignore_index=True)
    summary_results = pd.DataFrame([summary.__dict__ for summary in summaries])

    distribution_results.to_csv(results_dir / "distribution_metrics.csv", index=False)
    tstr_results.to_csv(results_dir / "tstr_metrics.csv", index=False)
    summary_results.to_csv(results_dir / "utility_summary.csv", index=False)
    return summary_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    summary = run_utility_evaluation(args.config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
