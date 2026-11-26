#!/usr/bin/env bash
# Run one member's part of the MP074 project, then optionally commit only their files.
#
#   scripts/run_part.sh data      # Data lead:    DVC hashes + seven data checks
#   scripts/run_part.sh model     # Model lead:   full pipeline (MLflow, registry, holdout)
#   scripts/run_part.sh service   # Service lead: pytest + Docker build (if Docker is running)
#   scripts/run_part.sh ops       # Ops lead:     CI and Kubernetes checks + monitoring results
#
# Add --commit to stage and commit only your role's files with YOUR git identity.
# Add --push (with --commit) to pull --rebase and push to origin.
# The script never sets a git identity: each member commits from their own account.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROLE="${1:-}"
shift || true
COMMIT=0
PUSH=0
for arg in "$@"; do
  case "$arg" in
    --commit) COMMIT=1 ;;
    --push) PUSH=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

# Python: the project venv if present, otherwise python3. Code under src/ is found via PYTHONPATH.
PY="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; fi
export PYTHONPATH="$ROOT/src"

# Files each member owns. Keep these lists in sync with the README roles table.
case "$ROLE" in
  data)
    FILES=(src/mp074/cleaning.py src/mp074/data_checks.py
           data/raw/train.csv.dvc data/raw/later.csv.dvc data/raw/holdout_features.csv data/raw/.gitignore
           .dvc/config .dvc/.gitignore .dvcignore .gitignore)
    ;;
  model)
    FILES=(src/mp074/models.py src/mp074/pipeline.py src/mp074/config.py src/mp074/__init__.py
           params.yaml requirements.txt mlflow.db models/model.joblib
           predictions/holdout_predictions.csv reports/findings.yaml)
    ;;
  service)
    FILES=(app/main.py src/mp074/schemas.py tests pytest.ini Dockerfile .dockerignore)
    ;;
  ops)
    FILES=(.github/workflows/ci.yml k8s/deployment.yaml k8s/service.yaml
           src/mp074/logging_setup.py src/mp074/monitoring.py src/mp074/metrics.py src/mp074/__main__.py)
    ;;
  *)
    echo "usage: scripts/run_part.sh <data|model|service|ops> [--commit] [--push]"
    exit 2
    ;;
esac

banner() { echo; echo "== $1 =="; }

run_data() {
  banner "Data lead: DVC hashes and seven data checks"
  "$PY" scripts/part_checks.py data
}

run_model() {
  banner "Model lead: full pipeline"
  # MLflow version numbers grow on every run, so start from a clean registry (v1, v2).
  rm -rf mlflow.db mlruns mlartifacts models/model.joblib reports/findings.yaml predictions/holdout_predictions.csv
  "$PY" -m mp074.pipeline
  banner "Model lead: results"
  "$PY" scripts/part_checks.py findings
}

run_service() {
  banner "Service lead: tests"
  "$PY" -m pytest -q
  banner "Service lead: Docker image"
  if docker info >/dev/null 2>&1; then
    docker build -t mp074-api:1.0 .
    echo "  built mp074-api:1.0"
  else
    echo "  Docker is not running here, skipping the build. Start Docker and rerun to build it."
  fi
}

run_ops() {
  banner "Ops lead: CI and Kubernetes files"
  "$PY" scripts/part_checks.py ops
  banner "Ops lead: monitoring results (from the model part's run)"
  "$PY" scripts/part_checks.py findings
}

commit_part() {
  banner "Committing ${ROLE} files"
  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "  not a git repo yet: run 'git init -b main' and add the remote first"; exit 1
  fi
  name="$(git config user.name || true)"
  email="$(git config user.email || true)"
  if [[ -z "$name" || -z "$email" ]]; then
    echo "  set YOUR git identity first (only you can commit as you):"
    echo "    git config user.name \"Your Name\""
    echo "    git config user.email \"your-github-email\""
    exit 1
  fi
  # Never commit on main: put this role's work on its own branch first.
  current="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$current" == "main" ]]; then
    git checkout -q -b "${ROLE}-lead"
    echo "  created branch ${ROLE}-lead (main is protected)"
  fi
  # Stage only this role's files that exist. Never stage the raw CSVs.
  git reset -q
  staged=0
  for f in "${FILES[@]}"; do
    if [[ -e "$f" ]]; then git add -- "$f"; staged=1; fi
  done
  git reset -q -- data/raw/train.csv data/raw/later.csv 2>/dev/null || true
  if [[ "$staged" -eq 0 ]] || git diff --cached --quiet; then
    echo "  nothing new to commit for ${ROLE}"; return
  fi
  git diff --cached --stat
  git commit -q -m "MP074 ${ROLE} part"
  echo "  committed as ${name} <${email}>"
  if [[ "$PUSH" -eq 1 ]]; then
    git pull -q --rebase
    git push -q -u origin HEAD
    echo "  pushed"
  fi
}

case "$ROLE" in
  data) run_data ;;
  model) run_model ;;
  service) run_service ;;
  ops) run_ops ;;
esac

if [[ "$COMMIT" -eq 1 ]]; then commit_part; fi

echo; echo "Done: ${ROLE} part."
