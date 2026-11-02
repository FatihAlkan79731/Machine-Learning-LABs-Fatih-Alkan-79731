"""Step 3: system cost of each model.

- training time: 1 warm-up fit, then median of 5 fits
- single-sample inference latency: 1 warm-up, then median of 100 predictions
- model size: joblib.dump to results/, size in bytes and KB
- peak memory: process RSS sampled with psutil while training and while predicting

Each model is measured in its own Python process so that the memory numbers
of one model do not leak into the other.
Usage:  python measure_cost.py            (measures all models)
        python measure_cost.py --model rf (measures a single model, prints JSON)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import joblib
import pandas as pd
import psutil

from common import MODEL_NAMES, RESULTS, load_split, make_model, set_seeds

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
MB = 1024 * 1024


class PeakRSS:
    """Samples the RSS of the current process in a background thread."""

    def __init__(self, interval=0.001):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def measure(key):
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    # training time
    make_model(key).fit(X_train, y_train)  # warm-up
    times = []
    with PeakRSS() as mem_train:
        for _ in range(N_TRAIN_RUNS):
            model = make_model(key)
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            times.append(time.perf_counter() - t0)

    # single-sample inference latency
    sample = X_test[:1]
    model.predict(sample)  # warm-up
    latencies = []
    with PeakRSS() as mem_infer:
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(sample)
            latencies.append(time.perf_counter() - t0)

    # model size
    path = RESULTS / f"model_{key}.joblib"
    joblib.dump(model, path)
    size = path.stat().st_size

    return {
        "model": MODEL_NAMES[key],
        "train_time_ms": statistics.median(times) * 1000,
        "infer_latency_ms": statistics.median(latencies) * 1000,
        "model_size_bytes": size,
        "model_size_kb": size / 1024,
        "rss_before_mb": rss_before / MB,
        "peak_rss_train_mb": mem_train.peak / MB,
        "peak_rss_infer_mb": mem_infer.peak / MB,
        "train_mem_increase_mb": (mem_train.peak - rss_before) / MB,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_NAMES))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.model:
        print(json.dumps(measure(args.model)))
        return

    rows = []
    for key in MODEL_NAMES:
        out = subprocess.run(
            [sys.executable, __file__, "--model", key],
            capture_output=True, text=True, check=True,
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "system_cost.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
