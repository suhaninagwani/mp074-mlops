"""Run the whole MP074 pipeline end to end.

    python -m mp074.pipeline        (run from the repository root, with PYTHONPATH=src)

Steps:
  1. read the three CSVs and run the seven data checks on the raw train file;
  2. train v1 candidates, each tracked in MLflow with data_sha256;
  3. register the best v1 (lowest CV MAE on train) as version 1 = champion;
  4. monitoring: PSI per column and v1's MAE per destination on later.csv;
  5. retrain v2 on train + first 70% of later, judge on the last 30%;
  6. apply the promotion rule, save models/model.joblib;
  7. predict the holdout with the champion and write findings.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.model_selection import KFold, cross_val_score

from mp074.cleaning import clean_features, file_sha256
from mp074.config import REPO_ROOT, load_params, resolve
from mp074.data_checks import run_all_checks
from mp074.findings import build_template
from mp074.metrics import mae
from mp074.models import build_pipeline, make_estimator
from mp074.monitoring import drift_report, error_by_group, promotion_decision, retrain_v2

log = logging.getLogger("mp074.pipeline")


def _plain(value: Any) -> Any:
    """Turn numpy and pandas scalars into plain Python so yaml.safe_dump accepts them."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def train_v1_candidates(params: dict, train: pd.DataFrame, later: pd.DataFrame,
                        data_sha: str) -> dict[str, Any]:
    """Track every candidate in MLflow and return the best one by CV MAE on train."""
    cfg = params["model"]
    target = params["data"]["target"]
    X, y = clean_features(train), train[target]
    X_later, y_later = clean_features(later), later[target]
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])

    best: dict[str, Any] | None = None
    for cand in cfg["candidates"]:
        with mlflow.start_run(run_name=f"v1-{cand['name']}") as run:
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("candidate", cand["name"])
            mlflow.log_param("model_kind", cand["kind"])
            mlflow.log_params({f"hp_{k}": v for k, v in cand["params"].items()})
            mlflow.set_tag("stage", "v1-train")

            def fresh():
                return build_pipeline(make_estimator(cand["kind"], cand["params"], cfg["random_state"]))

            cv_mae = float(-cross_val_score(fresh(), X, y, cv=cv,
                                            scoring="neg_mean_absolute_error").mean())
            model = fresh().fit(X, y)
            later_mae = mae(y_later, model.predict(X_later))

            mlflow.log_metric("cv_mae_train", cv_mae)
            mlflow.log_metric("mae_later", later_mae)
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
            log.info("v1 %s cv_mae_train=%.4f mae_later=%.4f", cand["name"], cv_mae, later_mae)

            if best is None or cv_mae < best["cv_mae_train"]:
                best = {"run_id": run.info.run_id, "candidate": cand, "model": model,
                        "cv_mae_train": cv_mae, "mae_later": later_mae}
    return best


def register(name: str, run_id: str) -> str:
    """Register the model logged in run_id. The first registration is version 1."""
    return str(mlflow.register_model(f"runs:/{run_id}/model", name).version)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params()
    paths, data_cfg, model_cfg = params["paths"], params["data"], params["model"]
    target, name = data_cfg["target"], model_cfg["registered_name"]

    train_raw = pd.read_csv(resolve(paths["train"]))
    later_raw = pd.read_csv(resolve(paths["later"]))
    holdout_raw = pd.read_csv(resolve(paths["holdout"]))
    data_sha = file_sha256(resolve(paths["train"]))
    log.info("loaded train=%d later=%d holdout=%d", len(train_raw), len(later_raw), len(holdout_raw))

    checks = run_all_checks(train_raw, later_raw, holdout_raw, data_cfg)

    mlflow.set_tracking_uri("sqlite:///" + str(REPO_ROOT / "mlflow.db"))
    mlflow.set_experiment(model_cfg["experiment_name"])
    client = MlflowClient()

    best = train_v1_candidates(params, train_raw, later_raw, data_sha)
    v1_version = register(name, best["run_id"])
    client.set_registered_model_alias(name, "champion", v1_version)
    v1_model = best["model"]

    drift = drift_report(train_raw, later_raw, params["drift"]["psi_bins"])
    errors = error_by_group(v1_model, later_raw, data_cfg["group_column"], target)

    v2 = retrain_v2(params, train_raw, later_raw, data_sha, best["candidate"], v1_model)
    v2_version = register(name, v2["run_id"])
    rule = promotion_decision(v2["v1_mae_last30"], v2["v2_mae_last30"],
                              params["promotion"]["min_relative_mae_drop"])
    if rule["promoted"]:
        client.set_registered_model_alias(name, "champion", v2_version)
        champion_model, champion_version = v2["model"], v2_version
    else:
        champion_model, champion_version = v1_model, v1_version
    log.info("promotion: reduction=%.4f promoted=%s champion=v%s",
             rule["relative_mae_reduction"], rule["promoted"], champion_version)

    resolve(paths["model"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, resolve(paths["model"]))

    # Predict the holdout with the registry's champion, and check it matches the saved file.
    loaded = mlflow.sklearn.load_model(f"models:/{name}@champion")
    X_hold = clean_features(holdout_raw)
    predictions = loaded.predict(X_hold)
    if not np.allclose(predictions, champion_model.predict(X_hold)):
        raise RuntimeError("registry champion does not match the saved model")
    out = pd.DataFrame({"record_id": holdout_raw["record_id"], "prediction": predictions.round(4)})
    resolve(paths["holdout_predictions"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(resolve(paths["holdout_predictions"]), index=False)

    internal = {
        "problem_id": "MP074",
        "data_problems": checks,
        "mlflow": {
            "v1_run_id": best["run_id"],
            "champion_version": int(champion_version),
        },
        "drift": drift,
        "error_by_group": errors,
        "retraining": {
            "v1_mae_last30pct": v2["v1_mae_last30"],
            "v2_mae_last30pct": v2["v2_mae_last30"],
            **rule,
        },
    }
    findings_path = resolve(paths["findings"])
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    template = build_template(internal, params["group_id"])
    with open(findings_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain(template), handle, sort_keys=False, allow_unicode=True)
    log.info("wrote %s", findings_path)


if __name__ == "__main__":
    main()
