"""Step 5: train a 2 -> 8 -> 1 network on make_moons with the engine and with PyTorch.

Both versions start from the *same* weights (the engine's initial weights are copied
into the torch model), use full-batch gradient descent with the same learning rate,
ReLU in the hidden layer, a sigmoid output and the mean binary cross-entropy loss.
If the engine is correct, the two loss curves must be identical up to rounding.

Timing: each full training run is repeated 5 times (after one untimed warm-up run)
and the median is reported. PyTorch is timed in float64 (same arithmetic as the
engine, which uses Python floats) and in float32 (its default).

Outputs:
  results/moons_loss_curves.csv   - per-epoch loss and accuracy of both implementations
  results/moons_timing.csv        - median and all 5 run times
  results/moons_training.png      - loss curves and decision boundary
"""
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

from common import RESULTS, SEED, set_seeds
from engine import Value
from engine_nn import MoonsMLP, bce

N_SAMPLES = 200
NOISE = 0.1
EPOCHS = 300
LR = 1.0
HIDDEN = 8
RUNS = 5


def get_data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    return X.astype(np.float64), y.astype(np.int64)


# ------------------------------------------------------------------- engine
def train_engine(X, y, record=True):
    model = MoonsMLP(hidden=HIDDEN, seed=SEED)
    init = model.state()
    params = model.parameters()
    xs = [[Value(v) for v in row] for row in X]      # inputs are constant leaves
    log, nodes = [], 0
    for epoch in range(1, EPOCHS + 1):
        probs = [model(x) for x in xs]
        loss = sum((bce(p, int(t)) for p, t in zip(probs, y)), Value(0.0)) * (1.0 / len(y))
        order = loss.backward()
        nodes = len(order)
        for p in params:
            p.data -= LR * p.grad
        if record:
            acc = np.mean([(p.data > 0.5) == t for p, t in zip(probs, y)])
            log.append({"epoch": epoch, "loss_engine": loss.data, "acc_engine": acc})
    return model, init, log, nodes


# ------------------------------------------------------------------- torch
def build_torch(init, dtype):
    import torch
    from torch import nn

    model = nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1), nn.Sigmoid()).to(dtype)
    w1, b1, w2, b2 = init
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(w1, dtype=dtype))
        model[0].bias.copy_(torch.tensor(b1, dtype=dtype))
        model[2].weight.copy_(torch.tensor(w2, dtype=dtype))
        model[2].bias.copy_(torch.tensor(b2, dtype=dtype))
    return model


def train_torch(X, y, init, dtype, record=True):
    import torch

    model = build_torch(init, dtype)
    xt = torch.tensor(X, dtype=dtype)
    yt = torch.tensor(y, dtype=dtype).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()
    log = []
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        p = model(xt)
        loss = loss_fn(p, yt)
        loss.backward()
        opt.step()
        if record:
            acc = ((p.detach() > 0.5).to(dtype) == yt).to(dtype).mean().item()
            log.append({"epoch": epoch, "loss_torch": loss.item(), "acc_torch": acc})
    return model, log


def timed(fn, runs=RUNS):
    fn()  # warm-up
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def main():
    set_seeds()
    X, y = get_data()
    engine_model, init, log_e, nodes = train_engine(X, y)
    print(f"engine: final loss {log_e[-1]['loss_engine']:.6f}, accuracy {log_e[-1]['acc_engine']:.3f}, "
          f"graph per step: {nodes} nodes")

    rows = []
    try:
        import torch

        torch.set_num_threads(torch.get_num_threads())
        _, log_t = train_torch(X, y, init, torch.float64)
        df = pd.DataFrame(log_e).merge(pd.DataFrame(log_t), on="epoch")
        df["abs_diff_loss"] = (df["loss_engine"] - df["loss_torch"]).abs()
        print(f"torch : final loss {log_t[-1]['loss_torch']:.6f}; max |loss diff| over "
              f"{EPOCHS} epochs = {df['abs_diff_loss'].max():.2e}")
        impls = {
            "engine (Python floats)": lambda: train_engine(X, y, record=False),
            "PyTorch float64": lambda: train_torch(X, y, init, torch.float64, record=False),
            "PyTorch float32": lambda: train_torch(X, y, init, torch.float32, record=False),
        }
        threads = torch.get_num_threads()
    except ImportError:
        print("WARNING: torch is not installed - only the engine is trained and timed")
        df = pd.DataFrame(log_e)
        impls = {"engine (Python floats)": lambda: train_engine(X, y, record=False)}
        threads = None

    for name, fn in impls.items():
        times = timed(fn)
        med = statistics.median(times)
        rows.append({"implementation": name, "median_s": med, "per_epoch_ms": 1000 * med / EPOCHS,
                     "runs_s": " ".join(f"{t:.4f}" for t in times), "torch_threads": threads})
        print(f"{name:<24} median of {RUNS}: {med:.4f} s ({1000 * med / EPOCHS:.3f} ms/epoch)")
    timing = pd.DataFrame(rows)
    timing["slowdown_vs_fastest"] = timing["median_s"] / timing["median_s"].min()
    timing.to_csv(RESULTS / "moons_timing.csv", index=False, float_format="%.6g")
    df.to_csv(RESULTS / "moons_loss_curves.csv", index=False, float_format="%.10g")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(df["epoch"], df["loss_engine"], lw=3, alpha=0.6, label="engine")
    if "loss_torch" in df:
        ax.plot(df["epoch"], df["loss_torch"], "k--", lw=1.2, label="PyTorch (float64)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean BCE loss")
    ax.set_title("Training loss, same initial weights")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 60),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 60))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([engine_model([Value(a), Value(b)]).data for a, b in grid]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.25, colors=["tab:blue", "tab:orange"])
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolors="k", linewidths=0.3)
    ax.set_title(f"Engine decision boundary (train acc {log_e[-1]['acc_engine']:.3f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "moons_training.png", dpi=130)
    print("\n" + timing.drop(columns="runs_s").to_string(index=False))


if __name__ == "__main__":
    main()
