import pandas as pd

from lossaware.privacy import evaluate_privacy, sample_frame


def test_sample_frame_limits_rows_reproducibly():
    frame = pd.DataFrame({"x": range(20)})

    sampled = sample_frame(frame, max_rows=5, random_seed=42)

    assert len(sampled) == 5
    assert sampled.equals(sample_frame(frame, max_rows=5, random_seed=42))


def test_evaluate_privacy_returns_expected_metrics():
    train = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "num": [10, 11, 12, 13, 14, 15],
            "cat": ["a", "b", "b", "c", "c", "a"],
            "target": [0, 1, 1, 0, 0, 1],
        }
    )
    synthetic = train.copy()

    result = evaluate_privacy(
        generator="copy",
        real_train=train,
        real_test=test,
        synthetic=synthetic,
        categorical_columns=["cat"],
        numerical_columns=["num"],
        target_column="target",
        max_real_rows=None,
        max_synthetic_rows=None,
        random_seed=42,
    )

    assert result.generator == "copy"
    assert result.exact_match_rate == 1.0
    assert result.dcr_min == 0.0
