"""Configuration helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG_PATH = Path("config/default.json")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a JSON experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_random_seed(seed: int) -> None:
    """Seed common local random generators used by the framework."""
    random.seed(seed)
    np.random.seed(seed)
