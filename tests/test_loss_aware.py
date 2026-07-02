import pandas as pd

from lossaware.loss_aware import rank_candidates


def make_tradeoff() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "generator": ["a", "b", "c"],
            "mean_numeric_ks": [0.1, 0.2, 0.3],
            "mean_categorical_tvd": [0.1, 0.2, 0.3],
            "correlation_frobenius": [5.0, 1.0, 9.0],
            "mean_tstr_f1_gap": [0.1, 0.3, 0.5],
            "mean_tstr_auc_gap": [0.1, 0.3, 0.5],
            "dcr_median": [2.0, 2.0, 0.5],
            "exact_match_rate": [0.0, 0.0, 0.1],
            "membership_advantage": [0.1, 0.1, 0.1],
        }
    )


def test_rank_candidates_applies_privacy_constraints():
    ranked = rank_candidates(
        tradeoff=make_tradeoff(),
        weights={
            "distribution_mismatch": 0.4,
            "correlation_distortion": 0.3,
            "downstream_degradation": 0.3,
        },
        privacy_config={
            "min_median_dcr": 1.0,
            "max_exact_match_rate": 0.0,
            "max_membership_advantage": 0.2,
        },
    )

    assert not ranked.loc[ranked["generator"] == "c", "privacy_feasible"].iloc[0]
    assert ranked.iloc[0]["generator"] == "a"


def test_rank_candidates_normalizes_weights():
    ranked = rank_candidates(
        tradeoff=make_tradeoff(),
        weights={
            "distribution_mismatch": 4,
            "correlation_distortion": 3,
            "downstream_degradation": 3,
        },
        privacy_config={"min_median_dcr": 0.0},
    )

    assert "composite_loss" in ranked.columns
