"""Configuration management."""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class EloConfig:
    k_factor: float = 32.0
    initial_rating: float = 1500.0
    glicko2_tau: float = 0.5


@dataclass
class ModelConfig:
    train_end: str = "2022-01-01"
    val_end: str = "2024-01-01"
    lgbm_params: dict = field(default_factory=dict)
    xgb_params: dict = field(default_factory=dict)
    neural_params: dict = field(default_factory=dict)


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(path)
    return cfg
