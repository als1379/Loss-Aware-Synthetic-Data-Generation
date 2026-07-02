from pathlib import Path

import pandas as pd

from lossaware.reporting import build_final_summary


def test_build_final_summary_writes_report(tmp_path):
    dataset = "demo"
    results = tmp_path / "results" / dataset
    reports = tmp_path / "reports"
    results.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": "logistic_regression",
                "accuracy": 0.8,
                "f1": 0.7,
                "roc_auc": 0.9,
            }
        ]
    ).to_csv(results / "baseline_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": "gaussian_copula",
                "mean_numeric_ks": 0.1,
                "mean_categorical_tvd": 0.2,
                "correlation_frobenius": 1.0,
                "mean_tstr_f1_gap": 0.1,
                "mean_tstr_auc_gap": 0.1,
            }
        ]
    ).to_csv(results / "utility_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": "gaussian_copula",
                "dcr_median": 2.0,
                "exact_match_rate": 0.0,
                "membership_advantage": 0.1,
            }
        ]
    ).to_csv(results / "privacy_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": "gaussian_copula",
                "mean_numeric_ks": 0.1,
                "mean_categorical_tvd": 0.2,
                "correlation_frobenius": 1.0,
                "mean_tstr_f1_gap": 0.1,
                "mean_tstr_auc_gap": 0.1,
                "dcr_median": 2.0,
                "exact_match_rate": 0.0,
                "membership_advantage": 0.1,
            }
        ]
    ).to_csv(results / "privacy_utility_tradeoff.csv", index=False)
    pd.DataFrame(
        [
            {
                "generator": "gaussian_copula",
                "selection_loss": 0.1,
                "composite_loss": 0.1,
            }
        ]
    ).to_csv(results / "loss_aware_ranking.csv", index=False)

    report = build_final_summary(tmp_path / "results", reports, dataset)

    assert report == reports / "final_summary.md"
    assert Path(report).exists()
    assert (results / "final_results_summary.csv").exists()
