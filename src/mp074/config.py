"""Load params.yaml and resolve the paths it names against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = REPO_ROOT / "params.yaml"


def load_params(path: Path | str = PARAMS_FILE) -> dict[str, Any]:
    """Read the settings file. Uses safe_load, as the handbook asks."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(relative: str) -> Path:
    """Turn a path from params.yaml into an absolute path under the repo root."""
    return REPO_ROOT / relative
