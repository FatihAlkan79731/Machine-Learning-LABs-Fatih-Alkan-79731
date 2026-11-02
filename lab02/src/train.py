"""Step 6: mini-batch SGD for every (learning rate, width) pair.

Each run: shuffle every epoch, batch size 32, 50 epochs, same seed for the
initial weights and the shuffling, so runs differ only in lr and width.

Outputs:
  results/training_log.csv  - per-epoch train loss, test loss, test accuracy
  results/final_accuracy.csv
  results/loss_curves.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, SEED, load_data, set_seeds
from mlp import MLP

LEARNING_RATES = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32


def accuracy(model, X, Y):
    return float((model.predict(X) == Y.argmax(axis=1)).mean())


def train_one(width, lr, data):
    X_train, X_test, Y_train, Y_test = data
    set_seeds()
    model = MLP(width, rng=np.random.default_rng(SEED))
    rng = np.random.default_rng(SEED)
    n = X_train.shape[0]
    log = []

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        batch_losses = []
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            batch_losses.append(model.loss(X_train[idx], Y_train[idx]))
            model.step(model.backward(Y_train[idx]), lr)

        row = {
            "width": width,
            "lr": lr,
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "test_loss": model.loss(X_test, Y_test),
            "test_acc": accuracy(model, X_test, Y_test),
        }
        log.append(row)
        print(f"W={width:<3} lr={lr:<6} epoch {epoch:>2}  "
              f"train_loss={row['train_loss']:.4f}  test_loss={row['test_loss']:.4f}  "
              f"test_acc={row['test_acc']:.4f}")
    return log


def plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            d = df[(df.width == width) & (df.lr == lr)]
            ax.plot(d.epoch, d.train_loss, label=f"lr = {lr}")
        ax.axhline(np.log(10), color="grey", ls="--", lw=0.8, label="ln(10)")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("training loss (cross-entropy)")
    axes[1].legend()
    fig.suptitle("Training loss per epoch, batch size 32")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)


def main():
    data = load_data()
    logs = []
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            logs += train_one(width, lr, data)

    df = pd.DataFrame(logs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "training_log.csv", index=False, float_format="%.6f")

    final = (df[df.epoch == EPOCHS][["width", "lr", "train_loss", "test_loss", "test_acc"]]
             .rename(columns={"train_loss": "final_train_loss", "test_loss": "final_test_loss",
                              "test_acc": "final_test_acc"}))
    final.to_csv(RESULTS / "final_accuracy.csv", index=False, float_format="%.4f")
    print("\n" + final.round(4).to_string(index=False))
    plot(df)


if __name__ == "__main__":
    main()
