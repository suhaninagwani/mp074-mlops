"""Point the API at the committed champion model, whatever the working directory."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODEL_PATH", str(REPO_ROOT / "models" / "model.joblib"))
