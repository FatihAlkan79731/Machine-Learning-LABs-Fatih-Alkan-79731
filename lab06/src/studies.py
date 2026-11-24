"""Helpers shared by the study scripts: run train.py in a fresh process and read its results.

Every training run is a separate process, so peak-memory numbers are not mixed between
runs and every run starts from the same seeds. Each run also becomes its own MLflow run.
"""
import json
import subprocess
import sys

import pandas as pd

from config import RESULTS

SRC = RESULTS.parent / "src"
HISTORY = RESULTS / "history"


def run_train(run_name, overrides=(), extra=(), tags=None):
    HISTORY.mkdir(parents=True, exist_ok=True)
    summary = HISTORY / f"{run_name}_summary.json"
    cmd = [sys.executable, str(SRC / "train.py"), "--run-name", run_name, "--summary-json", str(summary)]
    for o in overrides:
        cmd += ["--set", o]
    for k, v in (tags or {}).items():
        cmd += ["--tag", f"{k}={v}"]
    cmd += list(extra)
    print(f"\n>>> {run_name}: {' '.join(overrides)} {' '.join(extra)}", flush=True)
    subprocess.run(cmd, check=True, cwd=SRC)
    return json.loads(summary.read_text(encoding="utf-8"))


def history(run_name):
    return pd.read_csv(HISTORY / f"{run_name}_history.csv")


def log_to_mlflow(cfg, run_name, params=None, metrics=None, artifacts=(), tags=None):
    """Record a study that does not go through train.py (profiling, worker study) as an MLflow run."""
    import mlflow

    from config import ROOT, flatten

    mlflow.set_tracking_uri((ROOT / cfg.mlflow.tracking_uri).resolve().as_uri())
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**flatten(cfg.to_dict()), **(params or {})})
        mlflow.set_tags({"run_name": run_name, **(tags or {})})
        if metrics:
            mlflow.log_metrics(metrics)
        for a in artifacts:
            mlflow.log_artifact(str(a))
