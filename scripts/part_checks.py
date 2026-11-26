"""Checks used by scripts/run_part.sh. Run from the repo root with PYTHONPATH=src.

Subcommands:
  data      verify DVC hashes, then run the seven data checks on the raw train file
  findings  print the key results from reports/findings.yaml
  ops       validate .github/workflows/ci.yml and the Kubernetes files
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FAILED = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILED.append(label)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def cmd_data() -> None:
    print("DVC hashes (must match the original CSVs)")
    for raw, dvc in [("data/raw/train.csv", "data/raw/train.csv.dvc"),
                     ("data/raw/later.csv", "data/raw/later.csv.dvc")]:
        raw_path, dvc_path = ROOT / raw, ROOT / dvc
        if not raw_path.exists():
            check(False, f"{raw} is missing: copy it from the zip into data/raw/")
            continue
        recorded = yaml.safe_load(dvc_path.read_text())["outs"][0]["md5"]
        check(md5(raw_path) == recorded, f"{raw} matches {dvc}")
    check((ROOT / "data/raw/holdout_features.csv").exists(), "data/raw/holdout_features.csv present")

    if FAILED:
        return

    import pandas as pd
    from mp074.config import load_params
    from mp074.data_checks import run_all_checks

    params = load_params()
    train = pd.read_csv(ROOT / params["paths"]["train"])
    later = pd.read_csv(ROOT / params["paths"]["later"])
    holdout = pd.read_csv(ROOT / params["paths"]["holdout"])
    print("\nSeven data problems (from train.csv as given)")
    for kind, result in run_all_checks(train, later, holdout, params["data"]).items():
        if result["present"]:
            details = {k: v for k, v in result.items() if k not in ("present", "stats_per_group")}
            print(f"  PRESENT  {kind:18s} {details}")
        else:
            print(f"  absent   {kind}")


def cmd_findings() -> None:
    path = ROOT / "reports/findings.yaml"
    if not path.exists():
        print("  reports/findings.yaml not found: run the model part first")
        return
    f = yaml.safe_load(path.read_text())
    m = f["monitoring"]
    print(f"  group id              : {f['group_id']}")
    print(f"  champion version      : {f['tracking']['champion_version']} (best run {f['tracking']['best_run_id']})")
    print(f"  most drifted feature  : {m['most_drifted_feature']} (PSI {m['psi'][m['most_drifted_feature']]:.3f})")
    print(f"  worst-served group    : {m['worst_served_group']}")
    print(f"  MAE v1 / v2 last 30%  : {m['mae_v1_on_check']:.4f} / {m['mae_v2_on_check']:.4f}  decision={m['decision']}")


def cmd_ops() -> None:
    print("GitHub Actions (ci.yml)")
    ci = ROOT / ".github/workflows/ci.yml"
    text = ci.read_text() if ci.exists() else ""
    check(any(line.rstrip() == "on:" for line in text.splitlines()), "key is 'on:' (not 'true:')")
    check("push" in text, "triggers on push")
    check("pytest" in text and "requirements.txt" in text, "installs requirements.txt and runs pytest")

    print("Kubernetes")
    dep = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text())
    svc = yaml.safe_load((ROOT / "k8s/service.yaml").read_text())
    spec = dep["spec"]
    container = spec["template"]["spec"]["containers"][0]
    labels = spec["template"]["metadata"]["labels"]
    check(dep["metadata"]["name"] == "mp074-api", "Deployment name is mp074-api")
    check(spec["replicas"] == 4, "replicas is 4")
    check(container["image"] == "mp074-api:1.0", "image is mp074-api:1.0")
    check(container["ports"][0]["containerPort"] == 8000, "container port 8000")
    check(spec["selector"]["matchLabels"].items() <= labels.items(), "Deployment selector matches pod labels")
    readiness, liveness = container["readinessProbe"], container["livenessProbe"]
    check(readiness["httpGet"]["path"] == "/health" and liveness["httpGet"]["path"] == "/health",
          "readiness and liveness both on /health")
    check(liveness["initialDelaySeconds"] > readiness["initialDelaySeconds"], "liveness waits longer")
    check(svc["metadata"]["name"] == "mp074-svc", "Service name is mp074-svc")
    port = svc["spec"]["ports"][0]
    check(port["port"] == 80 and port["targetPort"] == 8000, "Service port 80 -> 8000")
    check(svc["spec"]["selector"].items() <= labels.items(), "Service selector matches pod labels")


def main() -> None:
    commands = {"data": cmd_data, "findings": cmd_findings, "ops": cmd_ops}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(f"usage: part_checks.py {{{'|'.join(commands)}}}")
    commands[sys.argv[1]]()
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
