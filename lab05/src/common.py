"""Shared helpers for Lab 5: paths, seeds and timing."""
import random
import statistics
import time
from pathlib import Path

import numpy as np

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
RESULTS = LAB_DIR / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def time_call(fn, warmup=10, reps=100):
    """Call `fn` `warmup` times untimed, then `reps` times; return (median, all times) in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times
