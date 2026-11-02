"""Run every step of Lab 2 in order."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["print_versions.py", "sanity_checks.py", "train.py", "timing.py"]

for step in STEPS:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
