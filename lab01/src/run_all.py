"""Run every step of Lab 1 in order."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["print_versions.py", "train_baselines.py", "measure_cost.py", "budget_fit.py"]

for step in STEPS:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
