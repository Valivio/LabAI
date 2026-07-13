from __future__ import annotations

import platform
import sys
from pathlib import Path

import mlx.core as mx

from labai.core import config


def _check_path(name: str, value: str | None) -> tuple[bool, str]:
    if not value:
        return False, f"{name}: not configured"

    path = Path(value).expanduser()
    if not path.exists():
        return False, f"{name}: missing ({path})"

    return True, f"{name}: OK ({path})"


def doctor() -> int:
    checks: list[tuple[bool, str]] = []

    checks.append((sys.version_info >= (3, 13), f"Python: {platform.python_version()}"))
    checks.append((mx.default_device().type == mx.gpu, f"MLX device: {mx.default_device()}"))
    checks.append(_check_path(config.DATA_DIR_ENV_VAR, config.get_data_dir()))
    checks.append(_check_path(config.MODELS_DIR_ENV_VAR, config.get_models_dir()))

    print("LabAI doctor\n")

    for ok, message in checks:
        marker = "✓" if ok else "✗"
        print(f"{marker} {message}")

    return 0 if all(ok for ok, _ in checks) else 1
