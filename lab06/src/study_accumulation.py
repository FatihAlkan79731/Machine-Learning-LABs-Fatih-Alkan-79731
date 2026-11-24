"""Step 6: gradient accumulation - emulate batch 256 with batch 64 x 4 accumulation steps.

1. Gradient check on one effective batch: the gradient of the first 256 training samples
   computed as one batch is compared with the sum of 4 weighted micro-batch gradients.
2. Two 5-epoch runs with Adam (lr 1e-3): batch 256 / accumulation 1 and batch 64 /
   accumulation 4. Because the shuffling generator draws the same permutation for both
   runs (RandomSampler does not depend on the batch size), every optimizer step sees the
   same 256 samples in both runs.

Outputs: results/accumulation_gradcheck.csv, results/accumulation_study.csv,
         results/accumulation_study.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from config import RESULTS, load_config
from studies import history, run_train
from train import MODELS, get_datasets, set_seeds

RUNS = {"batch256_accum1": ["data.batch_size=256", "training.grad_accum_steps=1"],
        "batch64_accum4": ["data.batch_size=64", "training.grad_accum_steps=4"]}


def gradient_check():
    cfg = load_config()
    train, _, _ = get_datasets(cfg)
    x = torch.stack([train[i][0] for i in range(256)])
    y = torch.tensor([train[i][1] for i in range(256)])
    set_seeds(cfg.seed)
    model = MODELS[cfg.model.name]()
    loss_fn = nn.CrossEntropyLoss()

    model.zero_grad()
    loss_fn(model(x), y).backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    for k in range(4):
        xb, yb = x[64 * k:64 * (k + 1)], y[64 * k:64 * (k + 1)]
        (loss_fn(model(xb), yb) * (64 / 256)).backward()      # same weighting as train.py
    accum = [p.grad.clone() for p in model.parameters()]

    rows = []
    for (name, _), g1, g2 in zip(model.named_parameters(), full, accum):
        rows.append({"parameter": name, "numel": g1.numel(),
                     "max_abs_diff": (g1 - g2).abs().max().item(),
                     "max_abs_grad": g1.abs().max().item(),
                     "relative_diff": ((g1 - g2).norm() / g1.norm()).item()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "accumulation_gradcheck.csv", index=False, float_format="%.3e")
    print(df.to_string(index=False))


def main():
    gradient_check()
    rows = []
    for name, args in RUNS.items():
        s = run_train(name, args + ["training.epochs=5"], tags={"step": "6-grad-accumulation"})
        rows.append({"run": name, "micro_batch": s["batch_size"], "accum_steps": s["grad_accum_steps"],
                     "effective_batch": s["effective_batch"], "val_acc": s["final_val_acc"],
                     "test_acc": s["test_acc"], "train_time_s": s["train_time_s"],
                     "throughput_samples_s": s["mean_throughput_samples_s"],
                     "peak_memory_mb": s["peak_memory_mb"],
                     "training_memory_increase_mb": s["peak_memory_mb"] - s["memory_before_training_mb"]})
    h1, h2 = history("batch256_accum1"), history("batch64_accum4")
    df = pd.DataFrame(rows)
    df["max_abs_diff_train_loss_per_epoch"] = (h1["train_loss"] - h2["train_loss"]).abs().max()
    df["max_abs_diff_val_acc_per_epoch"] = (h1["val_acc"] - h2["val_acc"]).abs().max()
    df.to_csv(RESULTS / "accumulation_study.csv", index=False, float_format="%.6g")
    print("\n" + df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for h, name, style in [(h1, "batch 256", "o-"), (h2, "batch 64 × 4 accumulation", "x--")]:
        axes[0].plot(h["epoch"], h["train_loss"], style, label=name)
        axes[1].plot(h["epoch"], h["val_acc"], style, label=name)
    axes[0].set_ylabel("train loss")
    axes[1].set_ylabel("validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Gradient accumulation: effective batch 256")
    fig.tight_layout()
    fig.savefig(RESULTS / "accumulation_study.png", dpi=130)


if __name__ == "__main__":
    main()
