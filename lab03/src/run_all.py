"""Run every step of Lab 3 in order. Fashion-MNIST is downloaded to data/ on first run."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["print_versions.py", "cost.py", "train.py", "plot_tradeoffs.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
