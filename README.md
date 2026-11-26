# MP074 Parcel Delivery Days (Nagpur)

SCSE3040 MLOps mini project. Client: Nagpur Speed Parcel Hub.
Goal: predict `delivery_days` (regression, measured by MAE in days).

| Field | Value |
|---|---|
| Problem id | MP074 |
| Group id | TODO: fill in from the faculty e-mail |
| Batch / faculty | 13 / Ravi Pratap |

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU1122 | Apoorv Gupta |
| S24CSEU1123 | Suhani Nagwani |
| S24CSEU1095 | Aviral Verma |
| S24CSEU1103 | Ansh Pandey |

## Repository layout

| Path | What it is |
|---|---|
| `params.yaml` | All settings (paths, candidate models, drift bins, promotion rule). Read with `yaml.safe_load`. |
| `src/mp074/cleaning.py` | Cleaning rules shared by training, drift and the API. |
| `src/mp074/data_checks.py` | The seven data-problem checks, run on the raw `train.csv`. |
| `src/mp074/models.py` | Candidate regressors and the preprocessing pipeline. |
| `src/mp074/metrics.py` | PSI, MAE and MAE per group. |
| `src/mp074/pipeline.py` | Runs the whole workflow and writes `reports/findings.yaml`. |
| `src/mp074/logging_setup.py` | JSON logging to stdout. |
| `src/mp074/schemas.py` | `POST /predict` request schema with the client's two rules. |
| `app/main.py` | FastAPI service: `POST /predict`, `GET /health`. |
| `tests/` | pytest suite (17 tests, no CSVs needed; one uses `TestClient`). |
| `data/raw/*.dvc` | DVC pointers to `train.csv` and `later.csv`. The CSVs are not in git. |
| `mlflow.db` | MLflow tracking database with the runs and the registered model. |
| `models/model.joblib` | Champion model the service loads. |
| `predictions/holdout_predictions.csv` | Predictions for the September holdout. |
| `reports/findings.yaml` | Data findings, drift, group errors, promotion decision. |
| `Dockerfile`, `.dockerignore` | Container image `mp074-api:1.0`. |
| `.github/workflows/ci.yml` | Runs pytest on every push. |
| `k8s/` | Deployment `mp074-api` (4 replicas) and Service `mp074-svc`. |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src          # lets Python find the mp074 package under src/
python -m mp074.pipeline       # from the repository root: trains, tracks, promotes, predicts
pytest -q
uvicorn app.main:app --port 8000
```

Then `curl localhost:8000/health` and `curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"distance_km":1200,"weight_kg":12.5,"hub_backlog":2000,"pin_code_tier":2,"declared_value_rs":15000,"transfers":3,"service":"express","destination":"Tier-2"}'`.

## Key decisions

- `days_from_tracking_page` is excluded from the model: it is known only after delivery (leakage).
- `transfers = 9999` is a sentinel; it becomes missing and is filled with the training median.
- Spellings of `service` (18 raw forms) are mapped to 3 real levels.
- Champion v1 is chosen by 5-fold cross-validated MAE on train, not on `later.csv`.
- v2 is promoted only if its MAE on the last 30% of `later.csv` is at least 12% below v1's.

## Running your part

Each member runs only their own part with one script. Commands are run from the repo root.

```bash
scripts/run_part.sh data      # Data lead: DVC hashes + the seven data checks
scripts/run_part.sh model     # Model lead: full pipeline (MLflow, registry, holdout, findings)
scripts/run_part.sh service   # Service lead: pytest + Docker build (if Docker is running)
scripts/run_part.sh ops       # Ops lead: CI and Kubernetes checks + monitoring results
```

Add `--commit` to stage and commit only your role's files, and `--push` to also push. The script uses your own git identity (`git config user.name` / `user.email`) and refuses to run without one. It never stages the raw CSVs.

The model part rebuilds `mlflow.db`, so run it once, before the model lead commits `mlflow.db`.

## Re-running

MLflow version numbers grow with each run. For a clean registry (v1, v2), delete `mlflow.db`, `mlartifacts/` and `models/model.joblib` before running the pipeline again.
