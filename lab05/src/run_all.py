"""Run every step of Lab 5 in order."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["print_versions.py", "gradcheck.py", "train_moons.py", "torch_graph.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
