"""Shared helpers for Lab 7: data, training, evaluation and cost measurement."""
import gzip
import io
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
DATA = ROOT / "data"
RESULTS = LAB_DIR / "results"
MODELS_DIR = LAB_DIR / "models"          # trained weights, not tracked by git
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MEAN, STD = 0.2860, 0.3530
BATCH = 128
LR = 1e-3


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------- data
def load_tensors():
    """Fashion-MNIST as normalised float tensors (decoded once, kept in memory).

    55k train / 5k validation from the official training set (seeded permutation),
    official 10k test set.
    """
    tr = datasets.FashionMNIST(root=str(DATA), train=True, download=True)
    te = datasets.FashionMNIST(root=str(DATA), train=False, download=True)

    def prep(ds):
        x = ds.data.float().div_(255).sub_(MEAN).div_(STD).unsqueeze(1)
        return x, ds.targets.clone()

    x, y = prep(tr)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))
    tr_idx, va_idx = perm[:55_000], perm[55_000:]
    xt, yt = prep(te)
    return (x[tr_idx], y[tr_idx]), (x[va_idx], y[va_idx]), (xt, yt)


def loader(x, y, *extra, shuffle=False, batch=BATCH, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y, *extra), batch_size=batch, shuffle=shuffle, generator=g)


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def evaluate(model, x, y, batch=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x)


@torch.no_grad()
def predict_logits(model, x, batch=1000):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def train(model, train_data, val_data, epochs, lr=LR, masks=None, loss_fn=None, seed=SEED, tag=""):
    """Adam training. `masks` ({param: 0/1 tensor}) are re-applied after every optimizer
    step, so pruned weights stay exactly zero during fine-tuning.
    `loss_fn(logits, y, *extra)` may use extra tensors of the dataset (e.g. teacher logits).
    Returns the per-epoch history and the total training time in seconds.
    """
    loss_fn = loss_fn or (lambda out, y, *_: nn.functional.cross_entropy(out, y))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = loader(*train_data, shuffle=True, seed=seed)
    hist, total = [], 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, n = 0.0, 0
        for x, y, *extra in dl:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, *extra)
            loss.backward()
            opt.step()
            if masks:
                with torch.no_grad():
                    for p, m in masks.items():
                        p.mul_(m)
            run_loss += loss.item() * y.size(0)
            n += y.size(0)
        dt = time.perf_counter() - t0
        total += dt
        val_acc = evaluate(model, *val_data)
        hist.append({"epoch": epoch, "train_loss": run_loss / n, "val_acc": val_acc, "epoch_time_s": dt})
        print(f"  {tag} epoch {epoch}: loss {run_loss / n:.4f}  val_acc {val_acc:.4f}  {dt:.1f}s", flush=True)
    return hist, total


# --------------------------------------------------------------------- cost
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_nonzero(model):
    return sum(int((p != 0).sum()) for p in model.parameters())


def serialized_bytes(obj, compress=False):
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    return len(gzip.compress(data, compresslevel=9)) if compress else len(data)


def flops_per_image(model, density=None):
    """Multiply-add FLOPs (x2) of every Conv2d / Linear layer for one 1x28x28 image.

    density: optional {module: fraction of non-zero weights}; then the FLOPs of that layer
    are scaled by it ("useful" FLOPs if zeros were skipped).
    """
    total = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            f = 2 * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1] * out.numel()
        else:
            f = 2 * m.in_features * m.out_features
        total += f * (density.get(m, 1.0) if density else 1.0)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28))
    for h in hooks:
        h.remove()
    return int(total)


def latency_ms(fn, x, warmup=20, reps=200):
    """Median latency (ms) of fn(x) under no_grad after `warmup` calls."""
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(x)
            times.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(times)


def model_latency(model, x_test):
    model.eval()
    return {"latency_b1_ms": latency_ms(model, x_test[:1], warmup=20, reps=200),
            "latency_b256_ms": latency_ms(model, x_test[:256], warmup=5, reps=30)}


# --------------------------------------------------------------------- energy
class Energy:
    """CodeCarbon energy measurement (kWh) of a code block; NaN if CodeCarbon fails.

        with Energy() as e:
            ...
        e.kwh
    """

    def __init__(self):
        self.kwh = float("nan")
        self.tracker = None

    def __enter__(self):
        try:
            from codecarbon import OfflineEmissionsTracker

            self.tracker = OfflineEmissionsTracker(country_iso_code="POL", save_to_file=False,
                                                   log_level="error", measure_power_secs=1)
            self.tracker.start()
        except Exception as e:  # noqa: BLE001
            print(f"  (CodeCarbon not available: {e})")
            self.tracker = None
        return self

    def __exit__(self, *exc):
        if self.tracker is not None:
            try:
                self.tracker.stop()
                data = getattr(self.tracker, "final_emissions_data", None)
                self.kwh = float(data.energy_consumed) if data is not None else float("nan")
            except Exception as e:  # noqa: BLE001
                print(f"  (CodeCarbon stop failed: {e})")
        return False


def inference_energy(model, x, repeats=5):
    """kWh for `repeats` passes over x (batch 256), and the same per 1,000 images."""
    model.eval()
    with Energy() as e, torch.no_grad():
        for _ in range(repeats):
            for i in range(0, len(x), 256):
                model(x[i:i + 256])
    return e.kwh, e.kwh / (repeats * len(x)) * 1000
