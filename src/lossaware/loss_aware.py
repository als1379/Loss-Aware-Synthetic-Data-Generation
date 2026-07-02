"""Loss-aware synthetic dataset selection."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionResult:
    """Selected synthetic dataset and ranking artifacts."""

    selected_generator: str
    selection_status: str
    selected_path: Path
    ranking_path: Path
    metadata_path: Path


LOSS_COLUMNS = {
    "distribution_mismatch": ["mean_numeric_ks", "mean_categorical_tvd"],
    "correlation_distortion": ["correlation_frobenius"],
    "downstream_degradation": ["mean_tstr_f1_gap", "mean_tstr_auc_gap"],
}


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a metric to [0, 1], treating lower raw values as better."""
    minimum = series.min()
    maximum = series.max()
    if np.isclose(maximum, minimum):
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def add_loss_components(tradeoff: pd.DataFrame) -> pd.DataFrame:
    """Add normalized composite loss components to a trade-off table."""
    ranked = tradeoff.copy()
    for component, columns in LOSS_COLUMNS.items():
        normalized_columns = []
        for column in columns:
            normalized_column = f"normalized_{column}"
            ranked[normalized_column] = minmax_normalize(ranked[column])
            normalized_columns.append(normalized_column)
        ranked[component] = ranked[normalized_columns].mean(axis=1)
    return ranked


def apply_privacy_constraints(
    ranked: pd.DataFrame, privacy_config: dict[str, Any]
) -> pd.DataFrame:
    """Mark candidates satisfying configured privacy constraints."""
    feasible = pd.Series(True, index=ranked.index)

    min_median_dcr = privacy_config.get("min_median_dcr")
    if min_median_dcr is not None:
        feasible &= ranked["dcr_median"] >= min_median_dcr

    max_exact_match_rate = privacy_config.get("max_exact_match_rate")
    if max_exact_match_rate is not None:
        feasible &= ranked["exact_match_rate"] <= max_exact_match_rate

    max_membership_advantage = privacy_config.get("max_membership_advantage")
    if max_membership_advantage is not None:
        feasible &= ranked["membership_advantage"] <= max_membership_advantage

    ranked = ranked.copy()
    ranked["privacy_feasible"] = feasible
    return ranked


def rank_candidates(
    tradeoff: pd.DataFrame,
    weights: dict[str, float],
    privacy_config: dict[str, Any],
) -> pd.DataFrame:
    """Rank synthetic candidates by weighted loss subject to privacy constraints."""
    ranked = add_loss_components(tradeoff)
    ranked = apply_privacy_constraints(ranked, privacy_config)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        weights = {key: value / total_weight for key, value in weights.items()}

    ranked["composite_loss"] = (
        weights["distribution_mismatch"] * ranked["distribution_mismatch"]
        + weights["correlation_distortion"] * ranked["correlation_distortion"]
        + weights["downstream_degradation"] * ranked["downstream_degradation"]
    )
    ranked["selection_loss"] = ranked["composite_loss"].where(
        ranked["privacy_feasible"],
        np.inf,
    )
    return ranked.sort_values(
        by=["selection_loss", "composite_loss", "membership_advantage"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def select_loss_aware_candidate(
    tradeoff_path: str | Path,
    synthetic_dir: str | Path,
    output_dir: str | Path,
    selection_config: dict[str, Any],
) -> SelectionResult:
    """Select and copy the best candidate synthetic dataset."""
    tradeoff = pd.read_csv(tradeoff_path)
    ranked = rank_candidates(
        tradeoff=tradeoff,
        weights=selection_config["weights"],
        privacy_config=selection_config["privacy"],
    )
    feasible = ranked[ranked["privacy_feasible"]]
    selection_status = "privacy_feasible"
    if feasible.empty:
        if not selection_config.get("allow_best_effort_if_no_feasible", True):
            raise ValueError("No synthetic candidate satisfies the privacy constraints.")
        feasible = ranked.sort_values(
            by=["composite_loss", "membership_advantage"],
            ascending=[True, True],
        )
        selection_status = "best_effort_no_privacy_feasible_candidate"

    selected = feasible.iloc[0]
    generator = str(selected["generator"])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "loss_aware_ranking.csv"
    metadata_path = destination / "loss_aware_selection.json"
    selected_path = destination / "loss_aware_selected.csv"

    ranked.to_csv(ranking_path, index=False)
    shutil.copyfile(Path(synthetic_dir) / f"{generator}.csv", selected_path)
    metadata_path.write_text(
        json.dumps(
            {
                "selected_generator": generator,
                "selection_status": selection_status,
                "selected_path": str(selected_path),
                "weights": selection_config["weights"],
                "privacy_constraints": selection_config["privacy"],
                "selected_metrics": selected.replace({np.inf: None}).to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SelectionResult(
        selected_generator=generator,
        selection_status=selection_status,
        selected_path=selected_path,
        ranking_path=ranking_path,
        metadata_path=metadata_path,
    )
