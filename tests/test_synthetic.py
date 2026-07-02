import pandas as pd

from lossaware.synthetic import build_metadata, prepare_training_frame


def test_prepare_training_frame_samples_stratified():
    frame = pd.DataFrame(
        {
            "num": range(20),
            "cat": ["a", "b"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )

    prepared = prepare_training_frame(
        train_frame=frame,
        categorical_columns=["cat"],
        numerical_columns=["num"],
        target_column="target",
        fit_max_rows=10,
        random_seed=42,
    )

    assert len(prepared) == 10
    assert set(prepared["target"]) == {"0", "1"}
    assert prepared["cat"].dtype == "object"


def test_build_metadata_marks_schema_types():
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "cat": ["a", "b"],
            "target": [0, 1],
        }
    )

    metadata = build_metadata(
        frame,
        target_column="target",
        categorical_columns=["cat"],
        numerical_columns=["num"],
    )
    metadata_dict = metadata.to_dict()

    assert metadata_dict["columns"]["num"]["sdtype"] == "numerical"
    assert metadata_dict["columns"]["cat"]["sdtype"] == "categorical"
    assert metadata_dict["columns"]["target"]["sdtype"] == "categorical"
