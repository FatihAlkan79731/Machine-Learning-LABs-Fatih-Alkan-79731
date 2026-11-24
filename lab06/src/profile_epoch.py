"""Step 2: profile one training epoch with torch.profiler.

Each training step is split into four labelled regions with torch.profiler.record_function:
  data_loading (next(iterator)), forward (model + loss), backward (zero_grad + loss.backward()),
  optimizer_step (opt.step()).
The share of each region is read from the profiler (CPU time of the region, children included).
The same regions are also timed with time.perf_counter, once inside the profiled epoch and
once in a second epoch without the profiler, to see how much the profiler itself distorts
the result.

Outputs:
  results/profile_breakdown.csv   - seconds and share per phase (profiler / timers / no profiler)
  results/profile_top_ops.txt     - the 30 most expensive operators (self CPU time)
  results/profile_breakdown.png
"""
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from config import RESULTS, load_config
from studies import log_to_mlflow
from train import MODELS, build_optimizer, get_loaders, set_seeds

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def one_epoch(model, loader, opt, loss_fn):
    model.train()
    timers = dict.fromkeys(PHASES, 0.0)
    it = iter(loader)
    steps = 0
    t_epoch = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        with record_function("data_loading"):
            batch = next(it, None)
        t1 = time.perf_counter()
        if batch is None:
            break
        x, y = batch
        with record_function("forward"):
            loss = loss_fn(model(x), y)
        t2 = time.perf_counter()
        with record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        t3 = time.perf_counter()
        with record_function("optimizer_step"):
            opt.step()
        t4 = time.perf_counter()
        for k, a, b in zip(PHASES, (t0, t1, t2, t3), (t1, t2, t3, t4)):
            timers[k] += b - a
        steps += 1
    return timers, time.perf_counter() - t_epoch, steps


def main():
    cfg = load_config(overrides=sys.argv[1:])
    set_seeds(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader, _, _ = get_loaders(cfg, g)
    model = MODELS[cfg.model.name]()
    opt = build_optimizer(cfg, model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        prof_timers, prof_total, steps = one_epoch(model, train_loader, opt, loss_fn)
    plain_timers, plain_total, _ = one_epoch(model, train_loader, opt, loss_fn)

    events = {e.key: e for e in prof.key_averages()}
    prof_ms = {p: events[p].cpu_time_total / 1e3 if p in events else float("nan") for p in PHASES}
    total_prof_ms = sum(prof_ms.values())

    rows = []
    for p in PHASES:
        rows.append({"phase": p,
                     "profiler_ms": prof_ms[p], "profiler_share": prof_ms[p] / total_prof_ms,
                     "timer_in_profiled_epoch_s": prof_timers[p],
                     "timer_share_profiled": prof_timers[p] / prof_total,
                     "timer_no_profiler_s": plain_timers[p],
                     "timer_share_no_profiler": plain_timers[p] / plain_total})
    other_p = prof_total - sum(prof_timers.values())
    other_n = plain_total - sum(plain_timers.values())
    rows.append({"phase": "other (loop, bookkeeping)", "timer_in_profiled_epoch_s": other_p,
                 "timer_share_profiled": other_p / prof_total, "timer_no_profiler_s": other_n,
                 "timer_share_no_profiler": other_n / plain_total})
    rows.append({"phase": "TOTAL epoch", "profiler_ms": total_prof_ms, "profiler_share": 1.0,
                 "timer_in_profiled_epoch_s": prof_total, "timer_share_profiled": 1.0,
                 "timer_no_profiler_s": plain_total, "timer_share_no_profiler": 1.0})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "profile_breakdown.csv", index=False, float_format="%.4f")
    print(df.round(4).to_string(index=False))
    print(f"steps per epoch: {steps}, batch {cfg.data.batch_size}, num_workers {cfg.data.num_workers}, "
          f"torch threads {torch.get_num_threads()}")

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    header = (f"torch.profiler, one epoch, CNN, batch {cfg.data.batch_size}, "
              f"num_workers {cfg.data.num_workers}, {steps} steps, torch {torch.__version__}, "
              f"threads {torch.get_num_threads()}\n\n")
    (RESULTS / "profile_top_ops.txt").write_text(header + table + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = ["profiler", "timers (profiled epoch)", "timers (no profiler)"]
    cols = ["profiler_share", "timer_share_profiled", "timer_share_no_profiler"]
    sub = df[df["phase"].isin(PHASES)]
    left = [0.0] * 3
    for _, r in sub.iterrows():
        vals = [r[c] for c in cols]
        ax.barh(labels, vals, left=left, label=r["phase"])
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(left[i] + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of epoch time")
    ax.set_title(f"One training epoch, CNN, batch {cfg.data.batch_size}, num_workers {cfg.data.num_workers}")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(RESULTS / "profile_breakdown.png", dpi=130)

    metrics = {f"share_{p}": float(df.loc[df.phase == p, "profiler_share"].iloc[0]) for p in PHASES}
    metrics["epoch_time_profiled_s"] = prof_total
    metrics["epoch_time_no_profiler_s"] = plain_total
    log_to_mlflow(cfg, "profile_one_epoch", metrics=metrics, tags={"step": "2-profiling"},
                  artifacts=[RESULTS / "profile_breakdown.csv", RESULTS / "profile_top_ops.txt",
                             RESULTS / "profile_breakdown.png"])


if __name__ == "__main__":
    main()
