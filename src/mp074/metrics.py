"""Drift and error measures: PSI (population stability index) and MAE."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a reference sample and a new sample.

    Bin edges are quantiles of the reference (expected) sample, so each bin
    holds about the same share of the reference. Missing values are ignored.
    Rule of thumb: below 0.1 is stable, 0.1 to 0.25 is a small shift, above 0.25 is a large shift.
    """
    e = pd.to_numeric(expected, errors="coerce").dropna().to_numpy(dtype=float)
    a = pd.to_numeric(actual, errors="coerce").dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0.0, 1.0, bins + 1)))
    inner = edges[1:-1]
    n_bins = len(inner) + 1
    e_share = np.bincount(np.digitize(e, inner), minlength=n_bins) / e.size
    a_share = np.bincount(np.digitize(a, inner), minlength=n_bins) / a.size
    e_share = np.clip(e_share, _EPS, None)
    a_share = np.clip(a_share, _EPS, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean absolute error, in the same unit as the target (days)."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def group_mae(df: pd.DataFrame, group: str, y_true: str, y_pred: str) -> dict[str, float]:
    """MAE for each value of the group column, worst group first."""
    rows = {}
    for value, part in df.groupby(group):
        rows[str(value)] = mae(part[y_true], part[y_pred])
    return dict(sorted(rows.items(), key=lambda kv: kv[1], reverse=True))
