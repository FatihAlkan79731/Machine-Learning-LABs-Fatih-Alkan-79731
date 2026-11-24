"""Step 4: batch-size study, batch in [32, 64, 128, 256, 512], 5 epochs each.

Same CNN, Adam with the same learning rate (1e-3) for every batch size, num_workers 0.
Every run is a separate train.py process, so the peak memory belongs to that run only.

Outputs: results/batch_study.csv, results/batch_study.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS
from studies import history, run_train

BATCHES = [32, 64, 128, 256, 512]


def main():
    rows = []
    for b in BATCHES:
        s = run_train(f"batch_{b}", [f"data.batch_size={b}", "training.epochs=5"],
                      tags={"step": "4-batch-size"})
        rows.append({"batch_size": b, "steps_per_epoch": -(-55_000 // b),
                     "throughput_samples_s": s["mean_throughput_samples_s"],
                     "train_time_5_epochs_s": s["train_time_s"],
                     "peak_memory_mb": s["peak_memory_mb"],
                     "memory_before_training_mb": s["memory_before_training_mb"],
                     "training_memory_increase_mb": s["peak_memory_mb"] - s["memory_before_training_mb"],
                     "val_acc": s["final_val_acc"], "test_acc": s["test_acc"],
                     "final_train_loss": history(f"batch_{b}")["train_loss"].iloc[-1]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "batch_study.csv", index=False, float_format="%.4f")
    print("\n" + df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = [str(b) for b in df["batch_size"]]
    axes[0].bar(x, df["throughput_samples_s"], color="tab:blue")
    axes[0].set_title("Throughput (samples/s)")
    axes[1].bar(x, df["peak_memory_mb"], color="tab:orange", label="peak")
    axes[1].bar(x, df["memory_before_training_mb"], color="tab:gray", alpha=0.6, label="before training")
    axes[1].set_title("Process memory (MB)")
    axes[1].legend(fontsize=8)
    axes[2].plot(x, df["test_acc"], "o-", label="test")
    axes[2].plot(x, df["val_acc"], "s--", label="validation")
    axes[2].set_title("Accuracy after 5 epochs")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("batch size")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "batch_study.png", dpi=130)


if __name__ == "__main__":
    main()
