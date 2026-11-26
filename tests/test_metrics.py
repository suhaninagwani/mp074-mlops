"""PSI and MAE behave as the drift and error checks expect."""

import numpy as np
import pandas as pd

from mp074.metrics import group_mae, mae, psi


def test_psi_is_near_zero_for_the_same_sample():
    rng = np.random.default_rng(0)
    sample = pd.Series(rng.normal(size=2000))
    assert psi(sample, sample) < 1e-9


def test_psi_grows_when_the_distribution_moves():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(size=2000))
    shifted = pd.Series(rng.normal(loc=2.0, size=2000))
    assert psi(base, shifted) > 0.25


def test_group_mae_lists_worst_group_first():
    df = pd.DataFrame({
        "destination": ["Metro", "Metro", "Rural", "Rural"],
        "y": [10.0, 12.0, 5.0, 5.0],
        "p": [9.0, 9.0, 5.0, 5.5],
    })
    result = group_mae(df, "destination", "y", "p")
    assert list(result) == ["Metro", "Rural"]
    assert round(result["Metro"], 6) == 2.0
    assert mae([1, 2], [2, 4]) == 1.5
