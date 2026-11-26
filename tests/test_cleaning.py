"""Cleaning rules: spelling variants, the 9999 sentinel and the leak column."""

import numpy as np
import pandas as pd
import pytest

from mp074.cleaning import FEATURES, clean_features, normalise_service


def _row(**overrides):
    row = {
        "record_id": "R1", "date": "2026-01-02", "distance_km": 100, "weight_kg": 2.5,
        "hub_backlog": 50, "pin_code_tier": 2, "declared_value_rs": 900, "transfers": 3,
        "service": "standard", "destination": "Metro", "days_from_tracking_page": 4.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("raw, clean", [
    ("standard", "standard"), (" standard", "standard"), ("Sta.", "standard"),
    ("ECONOMY", "economy"), ("eco.", "economy"), ("Express ", "express"), ("exp.", "express"),
])
def test_service_spellings_map_to_three_levels(raw, clean):
    assert normalise_service(raw) == clean


def test_unknown_service_is_rejected():
    with pytest.raises(ValueError):
        normalise_service("overnight")


def test_sentinel_transfers_become_missing_and_leak_is_dropped():
    cleaned = clean_features(_row(transfers=9999, service=" Economy"))
    assert np.isnan(cleaned.loc[0, "transfers"])
    assert cleaned.loc[0, "service"] == "economy"
    assert list(cleaned.columns) == FEATURES
    assert "days_from_tracking_page" not in cleaned.columns
