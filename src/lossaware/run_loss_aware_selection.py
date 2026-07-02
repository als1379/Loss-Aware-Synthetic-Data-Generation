"""Run loss-aware synthetic candidate selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from lossaware.config import load_config
from lossaware.loss_aware import select_loss_aware_candidate


def run_loss_aware_selection(config_path: str | Path = "config/default.json"):
    """Select the lowest-loss candidate satisfying privacy constraints."""
    config = load_config(config_path)
    dataset_name = config["dataset"]["name"]
    results_dir = Path(config["paths"]["results_dir"]) / dataset_name
    synthetic_dir = Path(config["paths"]["synthetic_data_dir"]) / dataset_name

    return select_loss_aware_candidate(
        tradeoff_path=results_dir / "privacy_utility_tradeoff.csv",
        synthetic_dir=synthetic_dir,
        output_dir=results_dir,
        selection_config=config["loss_aware_selection"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    result = run_loss_aware_selection(args.config)
    print(f"selected_generator={result.selected_generator}")
    print(f"selection_status={result.selection_status}")
    print(f"selected_path={result.selected_path}")
    print(f"ranking_path={result.ranking_path}")


if __name__ == "__main__":
    main()
