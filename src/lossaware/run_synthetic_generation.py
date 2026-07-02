"""Generate synthetic datasets with SDV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lossaware.config import load_config, set_random_seed
from lossaware.datasets import get_loader_from_config
from lossaware.synthetic import generate_synthetic_datasets


def run_synthetic_generation(config_path: str | Path = "config/default.json"):
    """Generate synthetic data from the real training split."""
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    set_random_seed(seed)

    loader = get_loader_from_config(config)
    dataset_name = config["dataset"]["name"]
    processed_dir = Path(config["paths"]["processed_data_dir"]) / dataset_name
    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run baseline training before synthetic generation."
        )

    train_frame = pd.read_csv(train_path)
    schema_bundle = loader.load(config["paths"]["raw_data_dir"])
    results = generate_synthetic_datasets(
        train_frame=train_frame,
        dataset_name=dataset_name,
        target_column=config["dataset"]["target_column"],
        categorical_columns=schema_bundle.categorical_columns,
        numerical_columns=schema_bundle.numerical_columns,
        generator_config=config["synthetic_generation"],
        output_dir=config["paths"]["synthetic_data_dir"],
        random_seed=seed,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    results = run_synthetic_generation(args.config)
    for result in results:
        print(
            f"{result.generator}: wrote {result.rows} rows to "
            f"{result.output_path}"
        )


if __name__ == "__main__":
    main()
