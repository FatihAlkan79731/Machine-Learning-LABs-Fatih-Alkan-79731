"""Step 4: train each model for 5 epochs with Adam (lr 1e-3, batch 128).

Outputs:
  results/training_log.csv  - per-epoch train loss, validation accuracy, epoch time
  results/test_accuracy.csv - final test accuracy per model
"""
import time

import pandas as pd
import torch
from torch import nn

from common import DEVICE, RESULTS, get_loaders, set_seeds
from models import MODELS, build

EPOCHS = 5
LR = 1e-3


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(dim=1)
            correct += (pred == y.to(DEVICE)).sum().item()
            total += y.numel()
    return correct / total


def train_model(name):
    set_seeds()
    train_loader, val_loader, test_loader = get_loaders()
    model = build(name).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        epoch_time = time.perf_counter() - t0
        val_acc = evaluate(model, val_loader)
        log.append({"model": name, "epoch": epoch, "train_loss": running / n,
                    "val_acc": val_acc, "epoch_time_s": epoch_time})
        print(f"{name:<10} epoch {epoch}  loss={running / n:.4f}  val_acc={val_acc:.4f}  "
              f"time={epoch_time:.1f}s")

    test_acc = evaluate(model, test_loader)
    print(f"{name:<10} test_acc={test_acc:.4f}")
    return log, {"model": name, "test_acc": test_acc, "val_acc": log[-1]["val_acc"],
                 "train_time_s": sum(r["epoch_time_s"] for r in log)}


def main():
    logs, finals = [], []
    for name in MODELS:
        log, final = train_model(name)
        logs += log
        finals.append(final)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(RESULTS / "training_log.csv", index=False, float_format="%.4f")
    df = pd.DataFrame(finals)
    df.to_csv(RESULTS / "test_accuracy.csv", index=False, float_format="%.4f")
    print("\n" + df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
