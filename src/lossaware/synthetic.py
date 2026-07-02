"""Synthetic data generation with SDV single-table synthesizers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)


@dataclass(frozen=True)
class SyntheticResult:
    """Saved synthetic dataset information."""

    generator: str
    rows: int
    output_path: Path
    config_path: Path


def build_metadata(
    frame: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> SingleTableMetadata:
    """Create SDV metadata from the project schema."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(frame)

    for column in categorical_columns + [target_column]:
        if column in frame.columns:
            metadata.update_column(column_name=column, sdtype="categorical")

    for column in numerical_columns:
        if column in frame.columns:
            metadata.update_column(column_name=column, sdtype="numerical")

    return metadata


def make_synthesizer(
    name: str,
    metadata: SingleTableMetadata,
    settings: dict[str, Any],
):
    """Instantiate an SDV synthesizer by registry name."""
    if name == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    if name == "ctgan":
        return CTGANSynthesizer(
            metadata,
            epochs=settings.get("epochs", 300),
            batch_size=settings.get("batch_size", 500),
            enable_gpu=settings.get("enable_gpu", True),
            verbose=settings.get("verbose", False),
        )
    if name == "tvae":
        return TVAESynthesizer(
            metadata,
            epochs=settings.get("epochs", 300),
            batch_size=settings.get("batch_size", 500),
            enable_gpu=settings.get("enable_gpu", True),
            verbose=settings.get("verbose", False),
        )
    raise ValueError(f"Unsupported synthetic generator: {name}")


def prepare_training_frame(
    train_frame: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
    fit_max_rows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Prepare a reproducible real-data sample for synthesizer fitting."""
    frame = train_frame.copy()
    if fit_max_rows is not None and len(frame) > fit_max_rows:
        frame = (
            frame.groupby(target_column, group_keys=False)
            .sample(frac=fit_max_rows / len(frame), random_state=random_seed)
            .reset_index(drop=True)
        )

    for column in categorical_columns + [target_column]:
        if column in frame:
            frame[column] = frame[column].astype("object")

    for column in numerical_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def generate_synthetic_datasets(
    train_frame: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    generator_config: dict[str, Any],
    output_dir: str | Path,
    random_seed: int,
) -> list[SyntheticResult]:
    """Fit enabled synthesizers and save generated synthetic datasets."""
    sample_size = generator_config["sample_size"]
    fit_max_rows = generator_config.get("fit_max_rows")
    fit_frame = prepare_training_frame(
        train_frame=train_frame,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        target_column=target_column,
        fit_max_rows=fit_max_rows,
        random_seed=random_seed,
    )
    metadata = build_metadata(
        fit_frame,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )

    destination = Path(output_dir) / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "sdv_metadata.json"
    metadata.save_to_json(metadata_path)

    results: list[SyntheticResult] = []
    for name, settings in generator_config["generators"].items():
        if not settings.get("enabled", False):
            continue

        synthesizer = make_synthesizer(name, metadata, settings)
        if hasattr(synthesizer, "set_random_state"):
            synthesizer.set_random_state(random_seed)
        synthesizer.fit(fit_frame)
        synthetic = synthesizer.sample(num_rows=sample_size)

        output_path = destination / f"{name}.csv"
        config_path = destination / f"{name}_config.json"
        synthetic.to_csv(output_path, index=False)
        config_path.write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "generator": name,
                    "sample_size": sample_size,
                    "fit_rows": len(fit_frame),
                    "settings": settings,
                    "random_seed": random_seed,
                    "target_column": target_column,
                    "categorical_columns": categorical_columns,
                    "numerical_columns": numerical_columns,
                    "metadata_path": str(metadata_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        results.append(
            SyntheticResult(
                generator=name,
                rows=len(synthetic),
                output_path=output_path,
                config_path=config_path,
            )
        )

    return results
