"""Convert pipeline results into the findings.yaml template the marking tool reads.

Keys, nesting and null rules follow the template in the group's repository:
a data issue that is absent gets only `present: false`, every other value null.
"""

from __future__ import annotations

from typing import Any

# The monitoring block of the template lists only these six model inputs.
TEMPLATE_PSI_FEATURES = ["distance_km", "weight_kg", "hub_backlog",
                         "pin_code_tier", "declared_value_rs", "transfers"]
TEMPLATE_GROUPS = ["Metro", "Tier-2", "Rural"]


def _issue(found: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
    """Map one internal check result onto the template's keys; absent means nulls."""
    if not found.get("present"):
        return {"present": False, **{t: None for t in keys}}
    return {"present": True, **{t: found.get(src) for t, src in keys.items()}}


def build_template(internal: dict[str, Any], group_id: str) -> dict[str, Any]:
    checks = internal["data_problems"]
    mlflow_info = internal["mlflow"]
    drift = internal["drift"]["psi_by_column"]
    errors = internal["error_by_group"]["v1_mae_later_by_destination"]
    retrain = internal["retraining"]

    dup = checks["duplicate_rows"]
    data_issues = {
        "leaking_column": _issue(checks["leaking_column"], {"column": "column"}),
        "unit_mixup": _issue(checks["unit_mixup"],
                             {"column": "column", "group_value": "group_value", "factor": "factor"}),
        "sentinel_value": _issue(checks["sentinel_value"],
                                 {"column": "column", "value": "value", "count": "count"}),
        "duplicate_rows": {"present": True, "count": dup["extra_copies"]}
        if dup.get("present") else {"present": False, "count": None},
        "stuck_sensor": _issue(checks["stuck_sensor"],
                               {"column": "column", "start_date": "first_date", "end_date": "last_date"}),
        "target_typos": _issue(checks["target_typos"], {"count": "count"}),
        "spelling_variants": _issue(checks["spelling_variants"],
                                    {"column": "column", "true_levels": "true_levels"}),
    }

    psi = {c: float(drift[c]) for c in TEMPLATE_PSI_FEATURES}
    most_drifted = max(psi, key=psi.get)
    promoted = bool(retrain["promoted"])

    return {
        "group_id": group_id,
        "problem_id": internal["problem_id"],
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": "mp074-delivery-days",
            "best_run_id": mlflow_info["v1_run_id"],
            "registered_model_name": "mp074-delivery-days",
            "champion_version": mlflow_info["champion_version"],
        },
        "monitoring": {
            "psi": psi,
            "most_drifted_feature": most_drifted,
            "mae_by_group": {g: float(errors[g]) for g in TEMPLATE_GROUPS},
            "worst_served_group": max(TEMPLATE_GROUPS, key=lambda g: errors[g]),
            "mae_v1_on_check": float(retrain["v1_mae_last30pct"]),
            "mae_v2_on_check": float(retrain["v2_mae_last30pct"]),
            "decision": "promote" if promoted else "keep",
        },
    }
