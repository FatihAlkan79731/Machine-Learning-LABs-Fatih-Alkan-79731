"""Step 3: vary DataLoader num_workers in [0, 1, 2, 4, 8] and find the bottleneck.

For every setting (batch 128, same CNN and seed):
  * startup_s      - time until the first batch arrives (worker processes are started here)
  * data_only_s    - one full pass over the training loader with no model work (how fast
                     the input pipeline alone can deliver data)
  * epoch_s        - one real training epoch (data + forward + backward + optimizer step)
Reference lines:
  * compute_only_s - one training epoch on batches that were decoded in advance and kept in
                     memory (the time the model needs when data costs nothing)

If epoch_s is close to data_only_s, the input pipeline is the bottleneck; if it is close to
compute_only_s, the model computation is.

Outputs: results/workers_study.csv, results/workers_study.png (+ one MLflow run per setting)
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from config import RESULTS, load_config
from studies import log_to_mlflow
from train import MODELS, build_optimizer, get_loaders, set_seeds

WORKERS = [0, 1, 2, 4, 8]


def train_epoch(model, batches, opt, loss_fn):
    model.train()
    t0 = time.perf_counter()
    n = 0
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()
        n += y.size(0)
    return time.perf_counter() - t0, n


def measure(nw):
    cfg = load_config(overrides=[f"data.num_workers={nw}"])
    set_seeds(cfg.seed)
    loader, _, _ = get_loaders(cfg, torch.Generator().manual_seed(cfg.seed))

    # data only: iterate the loader, touch nothing
    t0 = time.perf_counter()
    it = iter(loader)
    next(it)
    startup = time.perf_counter() - t0
    for _ in it:
        pass
    data_only = time.perf_counter() - t0
    del it

    # a real training epoch
    model = MODELS[cfg.model.name]()
    opt = build_optimizer(cfg, model.parameters())
    epoch, n = train_epoch(model, loader, opt, nn.CrossEntropyLoss())
    return cfg, {"num_workers": nw, "startup_s": startup, "data_only_s": data_only,
                 "epoch_s": epoch, "throughput_samples_s": n / epoch,
                 "data_only_samples_s": n / data_only}


def main():
    rows = []
    for nw in WORKERS:
        cfg, row = measure(nw)
        rows.append(row)
        print(f"num_workers={nw}: startup {row['startup_s']:.2f}s  data-only {row['data_only_s']:.2f}s  "
              f"epoch {row['epoch_s']:.2f}s  ({row['throughput_samples_s']:,.0f} samples/s)", flush=True)
        log_to_mlflow(cfg, f"workers_{nw}", metrics={k: v for k, v in row.items() if k != "num_workers"},
                      tags={"step": "3-num-workers"})

    # compute-only reference: pre-decoded batches in memory
    cfg = load_config()
    set_seeds(cfg.seed)
    loader, _, _ = get_loaders(cfg, torch.Generator().manual_seed(cfg.seed))
    cached = [(x.clone(), y.clone()) for x, y in loader]
    set_seeds(cfg.seed)
    model = MODELS[cfg.model.name]()
    opt = build_optimizer(cfg, model.parameters())
    compute_only, n = train_epoch(model, cached, opt, nn.CrossEntropyLoss())
    print(f"compute only (pre-loaded batches): {compute_only:.2f}s")

    df = pd.DataFrame(rows)
    df["compute_only_s"] = compute_only
    df["data_share_estimate"] = 1 - df["compute_only_s"] / df["epoch_s"]
    df["torch_threads"] = torch.get_num_threads()
    df.to_csv(RESULTS / "workers_study.csv", index=False, float_format="%.4f")
    print("\n" + df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    x = [str(w) for w in df["num_workers"]]
    ax.plot(x, df["epoch_s"], "o-", label="training epoch")
    ax.plot(x, df["data_only_s"], "s--", label="data loading only")
    ax.axhline(compute_only, color="gray", ls=":", label="compute only (pre-loaded)")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("seconds per epoch")
    ax.set_title("Where is the bottleneck?")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(x, df["startup_s"], color="tab:red")
    ax.set_xlabel("num_workers")
    ax.set_ylabel("seconds until first batch")
    ax.set_title("Worker start-up cost")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"DataLoader workers, CNN, batch {cfg.data.batch_size}, {torch.get_num_threads()} torch threads")
    fig.tight_layout()
    fig.savefig(RESULTS / "workers_study.png", dpi=130)


if __name__ == "__main__":
    main()
