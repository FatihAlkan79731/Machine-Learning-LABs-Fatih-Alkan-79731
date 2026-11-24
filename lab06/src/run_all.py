"""Run every step of Lab 6 in order (about one hour on a laptop CPU).

Fashion-MNIST is read from data/ (downloaded on first use). Checkpoints go to
lab06/checkpoints/ and MLflow runs to lab06/mlruns/; both are ignored by git.
Single steps can be run on their own, e.g.  python lab06/src/study_batch.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ["print_versions.py"],
    ["train.py", "--run-name", "baseline", "--tag", "step=1-baseline"],   # step 1
    ["profile_epoch.py"],                                                 # step 2
    ["study_workers.py"],                                                 # step 3
    ["study_batch.py"],                                                   # step 4
    ["study_schedules.py"],                                               # step 5
    ["study_accumulation.py"],                                            # step 6
    ["study_resume.py"],                                                  # step 7
    ["export_mlflow.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== {' '.join(step)} ===", flush=True)
        subprocess.run([sys.executable, str(SRC / step[0]), *step[1:]], check=True, cwd=SRC)
