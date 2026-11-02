"""Step 7: time the forward and the backward pass separately.

For each width and two batch sizes (32 = training batch, 1257 = whole training
set): 1 warm-up, then 100 timed runs of each pass; the median is reported.
The backward pass is timed on the cache of a forward pass done just before,
so only the gradient computation is measured.

The matmul FLOP count (2*m*k*n per product) is written next to the timings:
  forward : x W1, a1 W2, a2 W3                         -> 3 products
  backward: dW3, da2, dW2, da1, dW1                    -> 5 products
"""
import statistics
import time

import numpy as np
import pandas as pd

from common import RESULTS, load_data, set_seeds
from mlp import MLP

WIDTHS = [16, 64]
N_RUNS = 100


def matmul_flops(B, W, n_in=64, n_out=10):
    fwd = 2 * B * (n_in * W + W * W + W * n_out)
    bwd = 2 * B * (W * n_out + W * n_out + W * W + W * W + n_in * W)
    return fwd, bwd


def time_passes(model, X, Y):
    model.forward(X)
    model.backward(Y)  # warm-up
    fwd, bwd = [], []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.forward(X)
        t1 = time.perf_counter()
        model.backward(Y)
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
    return statistics.median(fwd) * 1e6, statistics.median(bwd) * 1e6


def main():
    set_seeds()
    X_train, _, Y_train, _ = load_data()
    rows = []
    for width in WIDTHS:
        model = MLP(width, rng=np.random.default_rng(42))
        for B in [32, X_train.shape[0]]:
            f_us, b_us = time_passes(model, X_train[:B], Y_train[:B])
            f_fl, b_fl = matmul_flops(B, width)
            rows.append({
                "width": width,
                "batch": B,
                "forward_us": f_us,
                "backward_us": b_us,
                "backward_over_forward": b_us / f_us,
                "forward_matmul_flops": f_fl,
                "backward_matmul_flops": b_fl,
                "flop_ratio": b_fl / f_fl,
            })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "timing.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 200):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
