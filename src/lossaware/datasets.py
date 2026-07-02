"""Dataset registry and loader interface.

The interface keeps dataset-specific download and cleaning details out of the
evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """A loaded tabular dataset plus schema metadata."""

    name: str
    frame: pd.DataFrame
    target_column: str
    categorical_columns: list[str]
    numerical_columns: list[str]
    metadata: dict[str, Any]


class DatasetLoader(Protocol):
    """Protocol for pluggable dataset loaders."""

    name: str

    def load(self, data_dir: str | Path = "data/raw") -> DatasetBundle:
        """Load the dataset and return a normalized bundle."""


def _columns_by_kind(
    frame: pd.DataFrame, target_column: str
) -> tuple[list[str], list[str]]:
    feature_columns = [column for column in frame.columns if column != target_column]
    categorical_columns = [
        column
        for column in feature_columns
        if frame[column].dtype == "object" or str(frame[column].dtype) == "category"
    ]
    numerical_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    return categorical_columns, numerical_columns


class AdultLoader:
    """Load UCI Adult/Census Income using ucimlrepo."""

    name = "adult"
    target_column = "income"

    def load(self, data_dir: str | Path = "data/raw") -> DatasetBundle:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError(
                "AdultLoader requires ucimlrepo. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        adult = fetch_ucirepo(id=2)
        features = adult.data.features.copy()
        targets = adult.data.targets.copy()
        target = targets.columns[0]

        frame = features.copy()
        frame[self.target_column] = (
            targets[target].astype(str).str.strip().str.replace(".", "", regex=False)
        )
        frame = frame.replace("?", pd.NA)

        categorical_columns, numerical_columns = _columns_by_kind(frame, self.target_column)

        Path(data_dir).mkdir(parents=True, exist_ok=True)

        return DatasetBundle(
            name=self.name,
            frame=frame,
            target_column=self.target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            metadata={
                "source": "UCI Machine Learning Repository",
                "uci_id": 2,
                "task": "Binary classification: income >50K vs <=50K",
            },
        )


class FolktablesACSIncomeLoader:
    """Load ACSIncome from Folktables.

    Default settings intentionally match the Folktables quick-start examples.
    The ACS feature columns are integer-coded by the Census. We mark code-like
    variables as categorical so mixed-type utility metrics and SDV metadata can
    treat them appropriately in later phases.
    """

    name = "folktables_acs_income"
    target_column = "income_gt_50k"

    categorical_columns = [
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "SEX",
        "RAC1P",
    ]
    numerical_columns = ["AGEP", "WKHP"]

    def __init__(
        self,
        survey_year: str = "2018",
        horizon: str = "1-Year",
        survey: str = "person",
        states: list[str] | None = None,
    ) -> None:
        self.survey_year = survey_year
        self.horizon = horizon
        self.survey = survey
        self.states = states or ["CA"]

    def load(self, data_dir: str | Path = "data/raw") -> DatasetBundle:
        try:
            from folktables import ACSDataSource, ACSIncome
        except ImportError as exc:
            raise ImportError(
                "FolktablesACSIncomeLoader requires folktables. Install dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        data_source = ACSDataSource(
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey=self.survey,
            root_dir=str(data_dir),
        )
        acs_data = data_source.get_data(states=self.states, download=True)
        features, labels, group = ACSIncome.df_to_pandas(acs_data)

        frame = features.copy()
        frame[self.target_column] = labels.astype(int)

        return DatasetBundle(
            name=self.name,
            frame=frame,
            target_column=self.target_column,
            categorical_columns=self.categorical_columns.copy(),
            numerical_columns=self.numerical_columns.copy(),
            metadata={
                "source": "Folktables ACSIncome",
                "survey_year": self.survey_year,
                "horizon": self.horizon,
                "survey": self.survey,
                "states": self.states,
                "group_column": getattr(group, "name", "RAC1P"),
                "task": "Binary classification: personal income > $50K",
            },
        )


DATASET_LOADERS: dict[str, DatasetLoader] = {
    AdultLoader.name: AdultLoader(),
    FolktablesACSIncomeLoader.name: FolktablesACSIncomeLoader(),
}


CANDIDATE_DATASETS: dict[str, dict[str, Any]] = {
    "uci_bank_marketing": {
        "title": "Bank Marketing",
        "task": "Term-deposit subscription prediction",
        "why_harder": [
            "larger mixed-type business dataset",
            "real campaign process with temporal/contact-history variables",
            "compact enough for repeated CTGAN/TVAE experiments",
        ],
        "status": "candidate",
    },
    "uci_default_credit_card": {
        "title": "Default of Credit Card Clients",
        "task": "Credit-card default prediction",
        "why_harder": [
            "more domain-relevant than Adult",
            "moderate class imbalance",
            "compact enough for repeated CTGAN/TVAE experiments",
        ],
        "status": "candidate",
    },
}


def list_datasets(include_candidates: bool = True) -> dict[str, dict[str, Any]]:
    """List implemented loaders and not-yet-implemented candidate datasets."""
    implemented = {
        name: {"implemented": True, "loader": loader.__class__.__name__}
        for name, loader in DATASET_LOADERS.items()
    }
    if not include_candidates:
        return implemented

    candidates = {
        name: {"implemented": False, **metadata}
        for name, metadata in CANDIDATE_DATASETS.items()
    }
    return {**implemented, **candidates}


def get_loader(name: str) -> DatasetLoader:
    """Return a dataset loader by registry name."""
    try:
        return DATASET_LOADERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unknown or unimplemented dataset '{name}'. Available: {available}") from exc


def get_loader_from_config(config: dict[str, Any]) -> DatasetLoader:
    """Build a dataset loader from the experiment configuration."""
    dataset_config = config["dataset"]
    name = dataset_config["name"]
    if name == FolktablesACSIncomeLoader.name:
        folktables_config = dataset_config.get("folktables", {})
        return FolktablesACSIncomeLoader(**folktables_config)
    return get_loader(name)
