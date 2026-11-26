"""Monitoring and retraining: drift (PSI), error by group, v2 retrain and promotion rule.

Covers handbook section 7 (PROBLEM.md): PSI per numeric column, v1's MAE per
destination, retraining v2 on train plus the first 70% of later.csv, and the
client's promotion rule judged on the last 30%.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd

from mp074.cleaning import clean_features, clean_frame
from mp074.metrics import group_mae, mae, psi

# Numeric columns compared between cleaned train and later. The last two are
# not model inputs, but the handbook asks for every numeric column.
NUMERIC_DRIFT_COLUMNS = [
    "distance_km", "weight_kg", "hub_backlog", "pin_code_tier", "declared_value_rs",
    "transfers", "days_from_tracking_page", "delivery_days",
]


def drift_report(train: pd.DataFrame, later: pd.DataFrame, bins: int) -> dict[str, Any]:
    """PSI for every numeric column, the most drifted column, and its value."""
    train_clean, later_clean = clean_frame(train), clean_frame(later)
    columns = [c for c in NUMERIC_DRIFT_COLUMNS if c in train_clean and c in later_clean]
    psi_by_column = {c: psi(train_clean[c], later_clean[c], bins) for c in columns}
    most = max(psi_by_column, key=psi_by_column.get)
    return {
        "method": f"PSI, {bins} quantile bins from cleaned train",
        "psi_by_column": psi_by_column,
        "most_drifted_column": most,
        "most_drifted_psi": psi_by_column[most],
    }


def error_by_group(model, later: pd.DataFrame, group: str, target: str) -> dict[str, Any]:
    """MAE of a fitted model on later.csv, for each group value, worst first."""
    scored = later.assign(prediction=model.predict(clean_features(later)))
    per_group = group_mae(scored, group, target, "prediction")
    return {"v1_mae_later_by_destination": per_group, "worst_served_group": next(iter(per_group))}


def retrain_v2(params: dict, train: pd.DataFrame, later: pd.DataFrame, data_sha: str,
               candidate: dict[str, Any], v1_model) -> dict[str, Any]:
    """Retrain the same candidate on train + first 70% of later; judge on the last 30%."""
    from mp074.models import build_pipeline, make_estimator  # imported here so ops merges before model

    cfg = params["model"]
    target = params["data"]["target"]
    frac = params["promotion"]["retrain_fraction"]

    ordered = later.assign(_d=pd.to_datetime(later["date"]))
    ordered = ordered.sort_values(["_d", "record_id"]).drop(columns="_d").reset_index(drop=True)
    cut = int(round(len(ordered) * frac))
    first, last = ordered.iloc[:cut], ordered.iloc[cut:]

    train_all = pd.concat([train, first], ignore_index=True)
    model = build_pipeline(make_estimator(candidate["kind"], candidate["params"], cfg["random_state"]))
    model.fit(clean_features(train_all), train_all[target])

    X_last, y_last = clean_features(last), last[target]
    v2_mae_last = mae(y_last, model.predict(X_last))
    v1_mae_last = mae(y_last, v1_model.predict(X_last))

    with mlflow.start_run(run_name="v2-retrain") as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("candidate", candidate["name"])
        mlflow.log_param("model_kind", candidate["kind"])
        mlflow.log_param("train_rows", len(train_all))
        mlflow.log_param("later_rows_used", cut)
        mlflow.set_tag("stage", "v2-retrain")
        mlflow.log_metric("mae_last30_v1", v1_mae_last)
        mlflow.log_metric("mae_last30_v2", v2_mae_last)
        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    return {"run_id": run_id, "model": model, "train_rows": len(train_all), "later_rows_used": cut,
            "judge_rows": len(last), "v1_mae_last30": v1_mae_last, "v2_mae_last30": v2_mae_last}


def promotion_decision(v1_mae: float, v2_mae: float, min_relative_drop: float) -> dict[str, Any]:
    """Client rule: promote v2 only if its MAE is at least `min_relative_drop` lower."""
    reduction = (v1_mae - v2_mae) / v1_mae
    promoted = v2_mae <= (1 - min_relative_drop) * v1_mae
    return {
        "relative_mae_reduction": reduction,
        "rule": f"promote v2 only if its MAE is at least {min_relative_drop:.0%} lower than v1",
        "promoted": promoted,
        "decision": "promote v2 as @champion" if promoted else "keep v1 as @champion",
    }
