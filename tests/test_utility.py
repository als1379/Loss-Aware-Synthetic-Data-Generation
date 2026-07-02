import pandas as pd

from lossaware.preprocessing import SplitData
from lossaware.utility import (
    correlation_distance,
    distribution_metrics,
    evaluate_tstr,
    total_variation_distance,
)


def test_total_variation_distance_zero_for_matching_distribution():
    real = pd.Series(["a", "a", "b"])
    synthetic = pd.Series(["a", "a", "b"])

    assert total_variation_distance(real, synthetic) == 0.0


def test_distribution_metrics_returns_numeric_and_categorical_rows():
    real = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "cat": ["a", "a", "b", "b"],
            "target": [0, 0, 1, 1],
        }
    )
    synthetic = pd.DataFrame(
        {
            "num": [1, 2, 2, 5],
            "cat": ["a", "b", "b", "b"],
            "target": [0, 1, 1, 1],
        }
    )

    metrics = distribution_metrics(
        real,
        synthetic,
        generator="toy",
        categorical_columns=["cat"],
        numerical_columns=["num"],
        target_column="target",
    )

    assert set(metrics["type"]) == {"numerical", "categorical"}
    assert set(metrics["column"]) == {"num", "cat", "target"}


def test_correlation_distance_is_zero_for_identical_data():
    frame = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "cat": ["a", "a", "b", "b"],
            "target": [0, 0, 1, 1],
        }
    )

    distance = correlation_distance(
        frame,
        frame.copy(),
        categorical_columns=["cat"],
        numerical_columns=["num"],
        target_column="target",
    )

    assert distance == 0.0


def test_evaluate_tstr_returns_gap_columns():
    synthetic = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "a", "b", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    split = SplitData(
        X_train=synthetic.drop(columns=["target"]),
        X_test=synthetic.drop(columns=["target"]),
        y_train=synthetic["target"],
        y_test=synthetic["target"],
        target_column="target",
        categorical_columns=["cat"],
        numerical_columns=["num"],
    )
    baseline = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "accuracy": 1.0,
                "f1": 1.0,
                "roc_auc": 1.0,
            }
        ]
    )

    result = evaluate_tstr(
        generator="toy",
        synthetic_train=synthetic,
        real_split=split,
        model_names=["logistic_regression"],
        baseline_results=baseline,
        random_seed=42,
    )

    assert {"accuracy_gap", "f1_gap", "roc_auc_gap"}.issubset(result.columns)
