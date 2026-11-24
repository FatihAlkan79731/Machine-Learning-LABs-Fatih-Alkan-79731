"""Run every step of Lab 7 in order (roughly 1-1.5 hours on a laptop CPU).

Trained weights are written to lab07/models/ (ignored by git); every table and figure
goes to lab07/results/.
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "print_versions.py",
    "baseline.py",            # step 1
    "prune_unstructured.py",  # step 2: global and per-layer magnitude pruning
    "prune_structured.py",    # step 2: filter pruning by L2 norm
    "plot_pruning.py",        # step 2: sparsity curves
    "latency_size.py",        # step 3
    "distill.py",             # step 4
    "summary.py",             # step 5
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
