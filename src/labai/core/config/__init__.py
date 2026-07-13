from __future__ import annotations

import os

DATA_DIR_ENV_VAR = "LABAI_DATA_DIR"
MODELS_DIR_ENV_VAR = "LABAI_MODELS_DIR"


def get_data_dir() -> str | None:
    return os.getenv(DATA_DIR_ENV_VAR)


def get_models_dir() -> str | None:
    return os.getenv(MODELS_DIR_ENV_VAR)
