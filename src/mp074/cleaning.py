"""Cleaning rules shared by training, evaluation, the drift check and the API.

Everything that changes a raw value lives here, so the model never sees
a different version of the data at training time and at serving time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Columns the model is allowed to use. These are exactly the fields the
# client's POST /predict accepts. days_from_tracking_page is NOT here: it is
# only known after delivery, so it would leak the answer.
FEATURES_NUMERIC = [
    "distance_km",
    "weight_kg",
    "hub_backlog",
    "pin_code_tier",
    "declared_value_rs",
    "transfers",
]
FEATURES_CATEGORICAL = ["service", "destination"]
FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

# Raw spelling -> real category. Keys are lower case, stripped and without a
# trailing full stop (so "Sta." and " standard" both map to "standard").
SERVICE_SPELLINGS = {
    "standard": "standard",
    "sta": "standard",
    "economy": "economy",
    "eco": "economy",
    "express": "express",
    "exp": "express",
}
DESTINATIONS = ("Metro", "Tier-2", "Rural")
TRANSFERS_SENTINEL = 9999


def normalise_service(value: object) -> str:
    """Return the clean service name, or raise ValueError for an unknown one."""
    key = str(value).strip().lower().rstrip(".")
    if key not in SERVICE_SPELLINGS:
        raise ValueError(f"unknown service: {value!r}")
    return SERVICE_SPELLINGS[key]


def normalise_destination(value: object) -> str:
    """Return the destination exactly as the model was trained on it."""
    text = str(value).strip()
    if text not in DESTINATIONS:
        raise ValueError(f"unknown destination: {value!r}")
    return text


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model's input columns with the cleaning rules applied.

    * service spellings are mapped to the three real levels;
    * the transfers sentinel 9999 becomes a missing value, which the model
      pipeline fills with the training median.
    """
    out = df.copy()
    out["service"] = out["service"].map(normalise_service)
    out["destination"] = out["destination"].map(normalise_destination)
    transfers = out["transfers"].astype(float)
    out["transfers"] = transfers.mask(transfers == TRANSFERS_SENTINEL, np.nan)
    return out[FEATURES].reset_index(drop=True)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a full file (features plus target if present), keeping every column."""
    out = df.copy()
    out["service"] = out["service"].map(normalise_service)
    transfers = out["transfers"].astype(float)
    out["transfers"] = transfers.mask(transfers == TRANSFERS_SENTINEL, np.nan)
    out["date"] = pd.to_datetime(out["date"])
    return out


def file_sha256(path: Path | str) -> str:
    """SHA-256 of the file exactly as stored on disk (for the data_sha256 param)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
