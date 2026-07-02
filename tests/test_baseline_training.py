import pandas as pd

from lossaware.baselines import evaluate_baselines
from lossaware.datasets import DatasetBundle
from lossaware.preprocessing import clean_frame, split_dataset


def make_bundle() -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "age": [22, 25, 47, 52, 46, 56, 23, 31, 44, 60, 28, 39],
            "hours": [30, 40, 50, 45, 60, 35, 20, 40, 55, 38, 42, 48],
            "workclass": [
                "private",
                "private",
                "gov",
                "self",
                "gov",
                "self",
                "private",
                "gov",
                "self",
                "private",
                "?",
                "gov",
            ],
            "target": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    return DatasetBundle(
        name="toy",
        frame=frame,
        target_column="target",
        categorical_columns=["workclass"],
        numerical_columns=["age", "hours"],
        metadata={},
    )


def test_clean_frame_normalizes_missing_and_types():
    cleaned = clean_frame(make_bundle())

    assert cleaned["workclass"].isna().sum() == 1
    assert str(cleaned["workclass"].dtype) == "object"


def test_split_dataset_preserves_schema():
    split = split_dataset(make_bundle(), test_size=0.25, random_seed=42)

    assert split.X_train.shape[0] == 9
    assert split.X_test.shape[0] == 3
    assert split.categorical_columns == ["workclass"]


def test_evaluate_baselines_returns_metrics():
    split = split_dataset(make_bundle(), test_size=0.25, random_seed=42)
    results = evaluate_baselines(
        dataset_name="toy",
        split=split,
        model_names=["logistic_regression"],
        random_seed=42,
    )

    assert set(results.columns) == {"dataset", "model", "accuracy", "f1", "roc_auc"}
    assert results.loc[0, "model"] == "logistic_regression"
