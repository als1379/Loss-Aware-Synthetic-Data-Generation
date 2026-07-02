"""Privacy-risk indicators for synthetic tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PrivacySummary:
    """Summary privacy-risk metrics for one synthetic dataset."""

    generator: str
    dcr_min: float
    dcr_p01: float
    dcr_p05: float
    dcr_median: float
    dcr_mean: float
    exact_match_rate: float
    train_closer_rate: float
    membership_advantage: float
    median_train_to_synthetic_distance: float
    median_test_to_synthetic_distance: float


def sample_frame(frame: pd.DataFrame, max_rows: int | None, random_seed: int) -> pd.DataFrame:
    """Return a reproducible sample if a frame is larger than max_rows."""
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)


def build_distance_transformer(
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> ColumnTransformer:
    """Create a mixed-type transformer for nearest-neighbour distances."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def nearest_distances(reference, query) -> np.ndarray:
    """Distance from every query row to its closest reference row."""
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neighbors.fit(reference)
    distances, _ = neighbors.kneighbors(query)
    return distances.ravel()


def evaluate_privacy(
    generator: str,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
    max_real_rows: int | None,
    max_synthetic_rows: int | None,
    random_seed: int,
    exact_match_tolerance: float = 1e-9,
) -> PrivacySummary:
    """Compute distance-based and membership-risk indicators.

    DCR is measured from synthetic records to real training records. The simple
    membership-risk proxy compares how close synthetic records are to real train
    versus held-out real test records. Values above zero suggest synthetic data
    is closer to training records than expected from the held-out reference.
    """
    columns = numerical_columns + categorical_columns + [target_column]
    sampled_train = sample_frame(real_train[columns], max_real_rows, random_seed)
    sampled_test = sample_frame(real_test[columns], max_real_rows, random_seed + 1)
    sampled_synthetic = sample_frame(synthetic[columns], max_synthetic_rows, random_seed + 2)

    categorical_with_target = categorical_columns + [target_column]
    transformer = build_distance_transformer(categorical_with_target, numerical_columns)
    transformer.fit(pd.concat([sampled_train, sampled_test], ignore_index=True))

    train_encoded = transformer.transform(sampled_train)
    test_encoded = transformer.transform(sampled_test)
    synthetic_encoded = transformer.transform(sampled_synthetic)

    synthetic_to_train = nearest_distances(train_encoded, synthetic_encoded)
    synthetic_to_test = nearest_distances(test_encoded, synthetic_encoded)
    train_to_synthetic = nearest_distances(synthetic_encoded, train_encoded)
    test_to_synthetic = nearest_distances(synthetic_encoded, test_encoded)

    train_closer_rate = float(np.mean(synthetic_to_train < synthetic_to_test))
    membership_advantage = max(0.0, train_closer_rate - 0.5)

    return PrivacySummary(
        generator=generator,
        dcr_min=float(np.min(synthetic_to_train)),
        dcr_p01=float(np.quantile(synthetic_to_train, 0.01)),
        dcr_p05=float(np.quantile(synthetic_to_train, 0.05)),
        dcr_median=float(np.median(synthetic_to_train)),
        dcr_mean=float(np.mean(synthetic_to_train)),
        exact_match_rate=float(np.mean(synthetic_to_train <= exact_match_tolerance)),
        train_closer_rate=train_closer_rate,
        membership_advantage=membership_advantage,
        median_train_to_synthetic_distance=float(np.median(train_to_synthetic)),
        median_test_to_synthetic_distance=float(np.median(test_to_synthetic)),
    )


def save_privacy_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """Persist privacy-risk metrics."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
