"""Utility evaluation for synthetic tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from lossaware.baselines import build_classifier
from lossaware.preprocessing import SplitData, build_preprocessor


@dataclass(frozen=True)
class UtilitySummary:
    """High-level utility metrics for one synthetic dataset."""

    generator: str
    mean_numeric_ks: float
    mean_categorical_tvd: float
    correlation_frobenius: float
    mean_tstr_f1_gap: float
    mean_tstr_auc_gap: float


def total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    """Compute total variation distance between categorical distributions."""
    real_probs = real.astype("object").value_counts(normalize=True, dropna=False)
    synthetic_probs = synthetic.astype("object").value_counts(normalize=True, dropna=False)
    categories = real_probs.index.union(synthetic_probs.index)
    return 0.5 * float(
        np.abs(
            real_probs.reindex(categories, fill_value=0)
            - synthetic_probs.reindex(categories, fill_value=0)
        ).sum()
    )


def distribution_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    generator: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Compare per-feature real and synthetic distributions."""
    rows: list[dict[str, object]] = []
    for column in numerical_columns:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna()
        statistic, p_value = ks_2samp(real_values, synthetic_values)
        rows.append(
            {
                "generator": generator,
                "column": column,
                "type": "numerical",
                "metric": "ks_statistic",
                "value": statistic,
                "p_value": p_value,
            }
        )

    for column in categorical_columns + [target_column]:
        real_values = real[column].astype("object")
        synthetic_values = synthetic[column].astype("object")
        tvd = total_variation_distance(real_values, synthetic_values)
        categories = real_values.value_counts(dropna=False).index.union(
            synthetic_values.value_counts(dropna=False).index
        )
        observed = np.vstack(
            [
                real_values.value_counts(dropna=False).reindex(categories, fill_value=0),
                synthetic_values.value_counts(dropna=False).reindex(categories, fill_value=0),
            ]
        )
        try:
            _, p_value, _, _ = chi2_contingency(observed)
        except ValueError:
            p_value = np.nan

        rows.append(
            {
                "generator": generator,
                "column": column,
                "type": "categorical",
                "metric": "total_variation_distance",
                "value": tvd,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def association_matrix(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Build a simple mixed-data association matrix through one-hot encoding."""
    columns = numerical_columns + categorical_columns + [target_column]
    encoded = pd.get_dummies(frame[columns], columns=categorical_columns + [target_column])
    corr = encoded.corr(numeric_only=True).fillna(0.0)
    return corr


def correlation_distance(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
) -> float:
    """Compute Frobenius norm between real and synthetic association matrices."""
    real_corr = association_matrix(real, categorical_columns, numerical_columns, target_column)
    synthetic_corr = association_matrix(
        synthetic,
        categorical_columns,
        numerical_columns,
        target_column,
    )
    columns = real_corr.columns.union(synthetic_corr.columns)
    real_aligned = real_corr.reindex(index=columns, columns=columns, fill_value=0.0)
    synthetic_aligned = synthetic_corr.reindex(index=columns, columns=columns, fill_value=0.0)
    return float(np.linalg.norm(real_aligned.to_numpy() - synthetic_aligned.to_numpy(), ord="fro"))


def evaluate_tstr(
    generator: str,
    synthetic_train: pd.DataFrame,
    real_split: SplitData,
    model_names: list[str],
    baseline_results: pd.DataFrame,
    random_seed: int,
) -> pd.DataFrame:
    """Train on synthetic data and test on real held-out data."""
    X_train = synthetic_train.drop(columns=[real_split.target_column])
    y_train = synthetic_train[real_split.target_column]
    label_encoder = LabelEncoder()
    y_real_train = label_encoder.fit_transform(real_split.y_train)
    y_synthetic = label_encoder.transform(y_train)
    y_test = label_encoder.transform(real_split.y_test)

    rows: list[dict[str, object]] = []
    for model_name in model_names:
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        real_split.categorical_columns,
                        real_split.numerical_columns,
                    ),
                ),
                ("classifier", build_classifier(model_name, random_seed)),
            ]
        )
        pipeline.fit(X_train, y_synthetic)
        predictions = pipeline.predict(real_split.X_test)
        probabilities = pipeline.predict_proba(real_split.X_test)[:, 1]

        baseline = baseline_results.loc[baseline_results["model"] == model_name].iloc[0]
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        rows.append(
            {
                "generator": generator,
                "model": model_name,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": roc_auc,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_f1": baseline["f1"],
                "baseline_roc_auc": baseline["roc_auc"],
                "accuracy_gap": baseline["accuracy"] - accuracy,
                "f1_gap": baseline["f1"] - f1,
                "roc_auc_gap": baseline["roc_auc"] - roc_auc,
            }
        )

    return pd.DataFrame(rows)


def plot_feature_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    generator: str,
    categorical_columns: list[str],
    numerical_columns: list[str],
    target_column: str,
    output_dir: str | Path,
) -> None:
    """Save compact distribution comparison plots."""
    destination = Path(output_dir) / generator
    destination.mkdir(parents=True, exist_ok=True)

    for column in numerical_columns:
        plt.figure(figsize=(7, 4))
        sns.kdeplot(pd.to_numeric(real[column], errors="coerce"), label="real")
        sns.kdeplot(pd.to_numeric(synthetic[column], errors="coerce"), label="synthetic")
        plt.title(f"{generator}: {column}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(destination / f"{column}_distribution.png", dpi=150)
        plt.close()

    for column in categorical_columns + [target_column]:
        real_counts = real[column].astype("object").value_counts(normalize=True).head(12)
        synthetic_counts = (
            synthetic[column].astype("object").value_counts(normalize=True).reindex(real_counts.index, fill_value=0)
        )
        plot_frame = pd.DataFrame({"real": real_counts, "synthetic": synthetic_counts})
        plot_frame.plot(kind="bar", figsize=(8, 4))
        plt.title(f"{generator}: {column}")
        plt.tight_layout()
        plt.savefig(destination / f"{column}_distribution.png", dpi=150)
        plt.close()


def summarize_utility(
    generator: str,
    distribution: pd.DataFrame,
    correlation_frobenius: float,
    tstr: pd.DataFrame,
) -> UtilitySummary:
    """Create one summary row for a generator."""
    numeric = distribution[distribution["type"] == "numerical"]["value"]
    categorical = distribution[distribution["type"] == "categorical"]["value"]
    return UtilitySummary(
        generator=generator,
        mean_numeric_ks=float(numeric.mean()) if not numeric.empty else np.nan,
        mean_categorical_tvd=float(categorical.mean()) if not categorical.empty else np.nan,
        correlation_frobenius=correlation_frobenius,
        mean_tstr_f1_gap=float(tstr["f1_gap"].mean()),
        mean_tstr_auc_gap=float(tstr["roc_auc_gap"].mean()),
    )
