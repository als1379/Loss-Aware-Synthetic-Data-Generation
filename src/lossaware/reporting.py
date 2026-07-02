"""Final reporting helpers for the reproducible pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def build_final_summary(results_dir: str | Path, reports_dir: str | Path, dataset_name: str) -> Path:
    """Create a concise Markdown report from phase outputs."""
    result_root = Path(results_dir) / dataset_name
    report_root = Path(reports_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    baselines = pd.read_csv(result_root / "baseline_results.csv")
    utility = pd.read_csv(result_root / "utility_summary.csv")
    privacy = pd.read_csv(result_root / "privacy_metrics.csv")
    ranking = pd.read_csv(result_root / "loss_aware_ranking.csv")
    selection_metadata = json.loads(
        (result_root / "loss_aware_selection.json").read_text(encoding="utf-8")
    )
    selected_generator = selection_metadata["selected_generator"]
    selection_status = selection_metadata.get("selection_status", "privacy_feasible")
    selection = ranking[ranking["generator"] == selected_generator].iloc[0]
    tradeoff = pd.read_csv(result_root / "privacy_utility_tradeoff.csv")

    selected_tradeoff = tradeoff[tradeoff["generator"] == selected_generator].iloc[0]
    best_baseline = baselines.sort_values("f1", ascending=False).iloc[0]

    summary_table = tradeoff[
        [
            "generator",
            "mean_numeric_ks",
            "mean_categorical_tvd",
            "correlation_frobenius",
            "mean_tstr_f1_gap",
            "mean_tstr_auc_gap",
            "dcr_median",
            "exact_match_rate",
            "membership_advantage",
        ]
    ].copy()
    summary_table.to_csv(result_root / "final_results_summary.csv", index=False)

    report_path = report_root / "final_summary.md"
    report_path.write_text(
        "\n".join(
            [
                "# Loss-Aware Synthetic Data Generation Summary",
                "",
                "## Dataset",
                "",
                f"- Benchmark: `{dataset_name}`",
                "- Adult remains available as a simple baseline loader, but ACSIncome is the selected harder benchmark.",
                "",
                "## Real-Data Baseline",
                "",
                "The reference task is Train-Real-Test-Real classification.",
                "",
                f"- Best baseline model: `{best_baseline['model']}`",
                f"- Accuracy: {_format_float(best_baseline['accuracy'])}",
                f"- F1: {_format_float(best_baseline['f1'])}",
                f"- ROC AUC: {_format_float(best_baseline['roc_auc'])}",
                "",
                "## Synthetic Utility",
                "",
                "Utility was measured with distribution mismatch, correlation distortion, and Train-Synthetic-Test-Real performance gaps.",
                "",
                f"- Selected generator: `{selected_generator}`",
                f"- Mean numeric KS: {_format_float(selected_tradeoff['mean_numeric_ks'])}",
                f"- Mean categorical TVD: {_format_float(selected_tradeoff['mean_categorical_tvd'])}",
                f"- Correlation distance: {_format_float(selected_tradeoff['correlation_frobenius'])}",
                f"- Mean TSTR F1 gap: {_format_float(selected_tradeoff['mean_tstr_f1_gap'])}",
                f"- Mean TSTR AUC gap: {_format_float(selected_tradeoff['mean_tstr_auc_gap'])}",
                "",
                "## Privacy Indicators",
                "",
                "Privacy was measured with distance-to-closest-record and a simple membership-risk proxy.",
                "",
                f"- DCR median: {_format_float(selected_tradeoff['dcr_median'])}",
                f"- Exact match rate: {_format_float(selected_tradeoff['exact_match_rate'])}",
                f"- Membership advantage: {_format_float(selected_tradeoff['membership_advantage'])}",
                "",
                "## Loss-Aware Strategy",
                "",
                "The loss-aware selector normalized each utility metric, applied the configured composite-loss weights, and selected the lowest-loss candidate. If no candidate satisfies the privacy constraints, the report marks the selection as best-effort rather than hiding the failure.",
                "",
                f"- Selected candidate: `{selected_generator}`",
                f"- Selection status: `{selection_status}`",
                f"- Composite loss: {_format_float(selection['composite_loss'])}",
                f"- Privacy feasible: `{bool(selection['privacy_feasible'])}`",
                "",
                "## Ethics Connection",
                "",
                "This framework makes the privacy-utility trade-off explicit instead of treating synthetic data as automatically safe. Utility metrics show whether synthetic data remains useful for downstream ML, while privacy indicators check whether generation may be too close to real individuals. The loss-aware stage turns those measurements into an auditable decision rule: choose a useful synthetic dataset only when it also satisfies privacy constraints. That supports responsible data sharing, privacy-preserving AI, and more trustworthy ML experimentation.",
                "",
                "## Key Output Files",
                "",
                "- `results/folktables_acs_income/final_results_summary.csv`",
                "- `results/folktables_acs_income/loss_aware_ranking.csv`",
                "- `results/folktables_acs_income/loss_aware_selected.csv`",
                "- `results/folktables_acs_income/plots/distributions/`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path
