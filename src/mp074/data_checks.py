"""Find the seven kinds of data problem in train.csv.

Every answer comes from code looking at the file. Each check returns a dict
with `present` (True/False) and, when present, the details that
reports/findings.yaml asks for. All checks run on the RAW training file,
before any cleaning, because the point is to find what cleaning must fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

UNIT_FACTORS = (10, 12, 60, 100, 1000)  # km->m, years->months, hours->minutes, ...
SENTINEL_CANDIDATES = (-1, -999, 999, 9999, -9999)


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str,
                         threshold: float) -> dict[str, Any]:
    """A column missing from the holdout file and almost equal to the target."""
    for col in train.columns:
        if col in (target, "record_id", "date") or col in holdout.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        corr = float(train[col].corr(train[target]))
        if abs(corr) >= threshold:
            gap = float((train[col] - train[target]).abs().mean())
            return {
                "present": True,
                "column": col,
                "correlation_with_target": round(corr, 4),
                "mean_abs_difference_days": round(gap, 3),
                "missing_from_holdout": True,
            }
    return {"present": False}


def check_unit_mixup(train: pd.DataFrame, group: str, target: str,
                     ratio_threshold: float = 5.0) -> dict[str, Any]:
    """One group's median for a numeric column is many times the others."""
    cols = _numeric_columns(train, exclude={target, "record_id"})
    for col in cols:
        med = train.groupby(group)[col].median()
        if len(med) < 2 or med.min() <= 0:
            continue
        if float(med.max() / med.min()) < ratio_threshold:
            continue
        # The odd group is the one whose median sits furthest from the others.
        vs_others = {g: float(med[g] / med.drop(g).median()) for g in med.index}
        odd = max(vs_others, key=lambda g: max(vs_others[g], 1 / vs_others[g]))
        odd_ratio = max(vs_others[odd], 1 / vs_others[odd])
        factor = min(UNIT_FACTORS, key=lambda f: abs(f - odd_ratio))
        stats = train.groupby(group)[col].agg(["min", "mean", "max"]).round(3)
        return {
            "present": True,
            "column": col,
            "group_column": group,
            "group_value": str(odd),
            "factor": int(factor),
            "stats_per_group": {str(k): {m: float(v) for m, v in row.items()}
                                for k, row in stats.iterrows()},
        }
    return {"present": False}


def check_sentinel_value(train: pd.DataFrame, target: str, min_count: int = 5) -> dict[str, Any]:
    """A special number such as -1 or 9999 repeated many times in one column."""
    for col in _numeric_columns(train, exclude={target, "record_id"}):
        for value in SENTINEL_CANDIDATES:
            count = int((train[col] == value).sum())
            if count >= min_count:
                return {"present": True, "column": col, "value": value, "count": count}
    return {"present": False}


def check_duplicate_rows(train: pd.DataFrame) -> dict[str, Any]:
    """Extra copies of the same record (the id is ignored for the comparison)."""
    features = [c for c in train.columns if c != "record_id"]
    extra = int(train.duplicated(subset=features).sum())
    if extra == 0:
        return {"present": False, "extra_copies": 0}
    return {"present": True, "extra_copies": extra}


def check_stuck_sensor(train: pd.DataFrame, date_col: str, target: str,
                       min_days: int = 14, min_rows: int = 5) -> dict[str, Any]:
    """The same value on consecutive rows across a run of at least `min_days` days."""
    ordered = train.assign(_d=pd.to_datetime(train[date_col])).sort_values(["_d", "record_id"])
    for col in _numeric_columns(train, exclude={target, "record_id"}):
        values = ordered[col].to_numpy()
        run_id = np.concatenate([[0], np.cumsum(values[1:] != values[:-1])])
        runs = ordered.assign(_run=run_id).groupby("_run")
        for _, run in runs:
            if len(run) < min_rows:
                continue
            first, last = run["_d"].min(), run["_d"].max()
            span = (last - first).days + 1
            if span >= min_days:
                return {
                    "present": True,
                    "column": col,
                    "stuck_value": float(run[col].iloc[0]),
                    "first_date": first.strftime("%Y-%m-%d"),
                    "last_date": last.strftime("%Y-%m-%d"),
                    "days": int(span),
                }
    return {"present": False}


def check_target_typos(train: pd.DataFrame, target: str, top: int = 50,
                       gap_factor: float = 3.0) -> dict[str, Any]:
    """A small group of targets far above the rest, with an empty gap below them."""
    y = np.sort(train[target].to_numpy(dtype=float))
    median = float(np.median(y))
    tail = y[-top:]
    gaps = np.diff(tail)
    if gaps.size == 0:
        return {"present": False}
    k = int(np.argmax(gaps))
    if gaps[k] >= gap_factor * median:
        suspects = tail[k + 1:]
        return {
            "present": True,
            "count": int(suspects.size),
            "gap_below": round(float(gaps[k]), 3),
            "suspect_values": [round(float(v), 3) for v in suspects],
        }
    return {"present": False}


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame,
                            text_columns: list[str]) -> dict[str, Any]:
    """A text column where the same category is written several ways."""
    for col in text_columns:
        raw_levels = int(train[col].nunique())
        true_levels = int(later[col].nunique())
        if raw_levels > true_levels:
            true_set = set(later[col].unique())
            variant_rows = int((~train[col].isin(true_set)).sum())
            return {
                "present": True,
                "column": col,
                "raw_levels": raw_levels,
                "true_levels": true_levels,
                "rows_with_variant_spelling": variant_rows,
                "clean_levels": sorted(true_set),
            }
    return {"present": False}


def run_all_checks(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame,
                   data_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run the seven checks in the order of the handbook (section 7)."""
    target = data_cfg["target"]
    group = data_cfg["group_column"]
    return {
        "leaking_column": check_leaking_column(
            train, holdout, target, data_cfg["leak_correlation_threshold"]),
        "unit_mixup": check_unit_mixup(train, group, target),
        "sentinel_value": check_sentinel_value(train, target),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train, data_cfg["date"], target),
        "target_typos": check_target_typos(train, target),
        "spelling_variants": check_spelling_variants(train, later, ["service", "destination"]),
    }
