"""Real-data baseline classifiers for the downstream utility reference point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from lossaware.preprocessing import SplitData, build_preprocessor


@dataclass(frozen=True)
class BaselineResult:
    """Metrics from one real-data baseline model."""

    dataset: str
    model: str
    accuracy: float
    f1: float
    roc_auc: float


def build_classifier(name: str, random_seed: int):
    """Create a supported sklearn classifier."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=50,
            random_state=random_seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_seed, n_estimators=50)
    raise ValueError(f"Unsupported classifier: {name}")


def evaluate_baselines(
    dataset_name: str,
    split: SplitData,
    model_names: list[str],
    random_seed: int,
) -> pd.DataFrame:
    """Train real-data classifiers and return accuracy, F1, and ROC AUC."""
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(split.y_train)
    y_test = label_encoder.transform(split.y_test)

    results: list[BaselineResult] = []
    for model_name in model_names:
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        split.categorical_columns,
                        split.numerical_columns,
                    ),
                ),
                ("classifier", build_classifier(model_name, random_seed)),
            ]
        )
        pipeline.fit(split.X_train, y_train)
        predictions = pipeline.predict(split.X_test)
        probabilities = pipeline.predict_proba(split.X_test)[:, 1]

        results.append(
            BaselineResult(
                dataset=dataset_name,
                model=model_name,
                accuracy=accuracy_score(y_test, predictions),
                f1=f1_score(y_test, predictions),
                roc_auc=roc_auc_score(y_test, probabilities),
            )
        )

    return pd.DataFrame([result.__dict__ for result in results])


def save_baseline_results(results: pd.DataFrame, output_path: str | Path) -> None:
    """Persist baseline metrics as a CSV table."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
