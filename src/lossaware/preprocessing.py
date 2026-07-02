"""Preprocessing for real and synthetic tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lossaware.datasets import DatasetBundle


@dataclass(frozen=True)
class SplitData:
    """Train/test split with schema and target metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_column: str
    categorical_columns: list[str]
    numerical_columns: list[str]


def clean_frame(bundle: DatasetBundle) -> pd.DataFrame:
    """Apply dataset-agnostic cleaning before splitting.

    Missing markers are normalized to pandas NA. Categorical columns are stored
    as strings so integer-coded ACS columns are handled as discrete categories.
    Numeric columns are converted to numeric values when possible.
    """
    frame = bundle.frame.copy()
    frame = frame.replace(["?", " ?"], pd.NA)

    for column in bundle.categorical_columns:
        if column in frame:
            frame[column] = frame[column].astype("object").where(frame[column].notna(), None)

    for column in bundle.numerical_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def split_dataset(
    bundle: DatasetBundle,
    test_size: float,
    random_seed: int,
    max_rows: int | None = None,
) -> SplitData:
    """Create a stratified train/test split for a classification task."""
    frame = clean_frame(bundle)
    if max_rows is not None and len(frame) > max_rows:
        frame = (
            frame.groupby(bundle.target_column, group_keys=False)
            .sample(frac=max_rows / len(frame), random_state=random_seed)
            .reset_index(drop=True)
        )

    X = frame.drop(columns=[bundle.target_column])
    y = frame[bundle.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_column=bundle.target_column,
        categorical_columns=bundle.categorical_columns,
        numerical_columns=bundle.numerical_columns,
    )


def build_preprocessor(
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> ColumnTransformer:
    """Build an sklearn transformer for mixed tabular data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def save_split(split: SplitData, output_dir: str | Path) -> None:
    """Save cleaned real-data train/test splits for reproducibility."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    train = split.X_train.copy()
    train[split.target_column] = split.y_train
    test = split.X_test.copy()
    test[split.target_column] = split.y_test

    train.to_csv(destination / "train.csv", index=False)
    test.to_csv(destination / "test.csv", index=False)
