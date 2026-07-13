from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV_VAR = "LABAI_DATA_DIR"
MODELS_DIR_ENV_VAR = "LABAI_MODELS_DIR"


@dataclass(frozen=True)
class LabAIConfig:
    data_dir: Path | None
    models_dir: Path | None


def _normalize_path(value: str | None) -> Path | None:
    if not value:
        return None

    return Path(value).expanduser().absolute()


def load_config() -> LabAIConfig:
    return LabAIConfig(
        data_dir=_normalize_path(os.getenv(DATA_DIR_ENV_VAR)),
        models_dir=_normalize_path(os.getenv(MODELS_DIR_ENV_VAR)),
    )
