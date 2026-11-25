"""Config-driven training script for the Lab 3 CNN on Fashion-MNIST (Lab 6, step 1).

Everything is set in lab06/config.yaml; single keys can be changed on the command line:

    python lab06/src/train.py                                         # defaults
    python lab06/src/train.py --set optimizer.name=sgd --set optimizer.lr=0.05 --run-name sgd
    python lab06/src/train.py --stop-after 3 --run-name demo          # simulate an interruption
    python lab06/src/train.py --resume lab06/checkpoints/demo/epoch_03.pt

Features
  * seeds for random / numpy / torch and a seeded DataLoader generator
  * a checkpoint after every epoch with model, optimizer, scheduler, history and all RNG
    states (including the DataLoader's shuffling generator), so a resumed run continues
    exactly where it stopped
  * MLflow: all config values as params, per-epoch metrics, and the config, history,
    summary and final checkpoint as artifacts. A resumed run continues the same MLflow run.
  * gradient accumulation (training.grad_accum_steps) with exact per-sample weighting
  * LR schedules: none | step | cosine | plateau; optimizers: adam | sgd (+momentum)
"""
import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import RESULTS, ROOT, flatten, load_config


# ============================================================== building blocks
class CNN(nn.Module):
    """The best model of Lab 3 (test accuracy 0.892 after 5 epochs)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 x 14 x 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 x 7 x 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


MODELS = {"CNN": CNN}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets(cfg):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.data.normalize_mean,), (cfg.data.normalize_std,)),
    ])
    root = str(ROOT / cfg.data.root)
    full_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
    n_val = cfg.data.val_size
    train, val = random_split(full_train, [len(full_train) - n_val, n_val],
                              generator=torch.Generator().manual_seed(cfg.seed))
    return train, val, test


def get_loaders(cfg, generator):
    train, val, test = get_datasets(cfg)
    nw = cfg.data.num_workers
    common = dict(num_workers=nw, pin_memory=cfg.data.pin_memory,
                  persistent_workers=bool(cfg.data.persistent_workers and nw > 0))
    eval_bs = max(cfg.data.batch_size, 256)
    return (DataLoader(train, batch_size=cfg.data.batch_size, shuffle=True, generator=generator, **common),
            DataLoader(val, batch_size=eval_bs, shuffle=False, **common),
            DataLoader(test, batch_size=eval_bs, shuffle=False, **common))


def build_optimizer(cfg, params):
    o = cfg.optimizer
    if o.name == "adam":
        return torch.optim.Adam(params, lr=o.lr, weight_decay=o.weight_decay)
    if o.name == "sgd":
        return torch.optim.SGD(params, lr=o.lr, momentum=o.momentum, weight_decay=o.weight_decay)
    raise ValueError(f"unknown optimizer {o.name}")


def build_scheduler(cfg, opt):
    s = cfg.scheduler
    if s.name == "none":
        return None
    if s.name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s.step_size, gamma=s.gamma)
    if s.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s.t_max, eta_min=s.eta_min)
    if s.name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s.factor,
                                                          patience=s.patience)
    raise ValueError(f"unknown scheduler {s.name}")


def accumulation_groups(n_samples, batch_size, accum):
    """Size of every optimizer step's effective batch (the last group may be smaller)."""
    micro = [batch_size] * (n_samples // batch_size) + ([n_samples % batch_size] if n_samples % batch_size else [])
    return [sum(micro[i:i + accum]) for i in range(0, len(micro), accum)]


def peak_memory_mb():
    """Peak resident memory of this process (Windows: peak working set, Linux/macOS: max RSS)."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 2 ** 20
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 1024


def current_memory_mb():
    return psutil.Process().memory_info().rss / 2 ** 20


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, opt, loss_fn, device, accum, groups):
    """One pass over the training data. Returns (mean loss, accuracy, seconds, samples)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    n_micro = len(loader)
    loss_sum, correct, seen = 0.0, 0, 0
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)                        # mean over the micro-batch
        # weight = share of this micro-batch in its optimizer step, so the accumulated
        # gradient equals the gradient of the mean loss over the whole effective batch
        weight = y.size(0) / groups[i // accum]
        (loss * weight).backward()
        if (i + 1) % accum == 0 or i + 1 == n_micro:
            opt.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return loss_sum / seen, correct / seen, time.perf_counter() - t0, seen


# ============================================================== checkpoints
def save_checkpoint(path, **state):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])


# ============================================================== main routine
def run(cfg, run_name, stop_after=None, resume=None, summary_path=None, tags=None):
    device = torch.device(cfg.training.device)
    set_seeds(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader, val_loader, test_loader = get_loaders(cfg, generator)
    model = MODELS[cfg.model.name]().to(device)
    opt = build_optimizer(cfg, model.parameters())
    sched = build_scheduler(cfg, opt)
    loss_fn = nn.CrossEntropyLoss()
    accum = int(cfg.training.grad_accum_steps)
    groups = accumulation_groups(len(train_loader.dataset), cfg.data.batch_size, accum)
    history, start_epoch, mlflow_run_id = [], 1, None

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None and ckpt.get("scheduler") is not None:
            sched.load_state_dict(ckpt["scheduler"])
        restore_rng(ckpt["rng"], generator)
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        mlflow_run_id = ckpt.get("mlflow_run_id")
        run_name = ckpt.get("run_name", run_name)
        print(f"resumed from {resume} at epoch {start_epoch}")

    ckpt_dir = ROOT / cfg.checkpoint.dir / run_name
    mem_before = current_memory_mb()

    mlflow.set_tracking_uri((ROOT / cfg.mlflow.tracking_uri).resolve().as_uri())
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_id=mlflow_run_id, run_name=None if mlflow_run_id else run_name) as active:
        if mlflow_run_id is None:
            mlflow.log_params(flatten(cfg.to_dict()))
            mlflow.set_tags({"run_name": run_name, **(tags or {})})
            cfg_file = RESULTS / "history" / f"{run_name}_config.yaml"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            cfg_file.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
            mlflow.log_artifact(str(cfg_file))
        else:
            mlflow.set_tag("resumed_at_epoch", start_epoch)

        interrupted = False
        for epoch in range(start_epoch, cfg.training.epochs + 1):
            lr = opt.param_groups[0]["lr"]
            tr_loss, tr_acc, tr_time, seen = train_one_epoch(model, train_loader, opt, loss_fn,
                                                             device, accum, groups)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            if sched is not None:
                sched.step(val_loss) if cfg.scheduler.name == "plateau" else sched.step()
            row = {"epoch": epoch, "lr": lr, "train_loss": tr_loss, "train_acc": tr_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": tr_time,
                   "throughput_samples_s": seen / tr_time}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"[{run_name}] epoch {epoch:>2}  lr={lr:.2e}  train_loss={tr_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  {tr_time:.1f}s  "
                  f"{seen / tr_time:,.0f} samples/s", flush=True)

            if cfg.checkpoint.every_epoch:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:02d}.pt", epoch=epoch, run_name=run_name,
                                model=model.state_dict(), optimizer=opt.state_dict(),
                                scheduler=None if sched is None else sched.state_dict(),
                                rng=rng_state(generator), history=history, config=cfg.to_dict(),
                                mlflow_run_id=active.info.run_id)
            if stop_after is not None and epoch >= stop_after:
                interrupted = True
                print(f"[{run_name}] stopped after epoch {epoch} (simulated interruption)")
                break

        hist_file = RESULTS / "history" / f"{run_name}_history.csv"
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(hist_file, index=False, float_format="%.6f")
        mlflow.log_artifact(str(hist_file))

        summary = {"run_name": run_name, "mlflow_run_id": active.info.run_id,
                   "epochs_done": history[-1]["epoch"], "interrupted": interrupted,
                   "batch_size": cfg.data.batch_size, "grad_accum_steps": accum,
                   "effective_batch": cfg.data.batch_size * accum,
                   "optimizer": cfg.optimizer.name, "scheduler": cfg.scheduler.name,
                   "num_workers": cfg.data.num_workers,
                   "final_val_acc": history[-1]["val_acc"], "final_val_loss": history[-1]["val_loss"],
                   "best_val_acc": max(h["val_acc"] for h in history),
                   "train_time_s": sum(h["epoch_time_s"] for h in history),
                   "mean_throughput_samples_s": float(np.mean([h["throughput_samples_s"] for h in history])),
                   "memory_before_training_mb": mem_before, "peak_memory_mb": peak_memory_mb()}
        if not interrupted:
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            summary.update(test_loss=test_loss, test_acc=test_acc)
            mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
            print(f"[{run_name}] test_acc={test_acc:.4f}")
            if cfg.mlflow.log_model_checkpoint and cfg.checkpoint.every_epoch:
                mlflow.log_artifact(str(ckpt_dir / f"epoch_{history[-1]['epoch']:02d}.pt"))
        mlflow.log_metrics({"peak_memory_mb": summary["peak_memory_mb"]})
        sum_file = RESULTS / "history" / f"{run_name}_summary.json"
        sum_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(sum_file))
        if interrupted:
            mlflow.set_tag("status_note", f"interrupted after epoch {history[-1]['epoch']}")

    if summary_path:
        Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file (default lab06/config.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this epoch (interruption test)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    kwargs = {"overrides": args.set}
    if args.config:
        kwargs["path"] = args.config
    if args.resume:   # a resumed run uses the configuration stored in its checkpoint
        stored = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
        tmp = Path(args.resume).resolve().parent / "resume_config.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(stored), encoding="utf-8")
        kwargs["path"] = tmp
    cfg = load_config(**kwargs)
    tags = dict(t.split("=", 1) for t in args.tag)
    run(cfg, args.run_name, stop_after=args.stop_after, resume=args.resume,
        summary_path=args.summary_json, tags=tags)


if __name__ == "__main__":
    main()
