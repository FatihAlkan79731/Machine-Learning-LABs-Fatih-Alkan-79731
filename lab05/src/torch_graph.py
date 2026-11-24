"""Step 6: inspect PyTorch graphs, compare eager vs compiled/traced execution,
and measure the cost of non-contiguous tensors.

Part A - graph inspection of a small MLP (784 -> 128 -> 64 -> 10):
  * the module tree, the torch.fx graph, the autograd (grad_fn) graph built by the
    forward pass, and the TorchScript graph produced by torch.jit.trace
  * shape, dtype, stride and contiguity of the input, the parameters and some views
  -> results/graph_inspection.txt

Part B - eager vs compiled. Three models with a different balance of work:
  * tiny     : 2 -> 8 -> 1 (the make_moons network) - almost no arithmetic, pure overhead
  * mlp      : 784 -> 512 -> 512 -> 10              - dominated by matrix multiplications
  * pointwise: a chain of 10 element-wise ops on 4096 features (GELU/SiLU-like) - memory bound
  modes: eager, torch.jit.trace, torch.jit.trace + freeze, torch.jit.script, torch.compile.
  torch.compile is tried first; if it is not supported on this platform the error is
  recorded and only the TorchScript modes are measured.
  For every mode: time of the first call (includes tracing/compilation), then 20 warm-up
  calls, then the median of 200 (batch 1) or 50 (batch 256) calls under torch.no_grad().
  -> results/eager_vs_compiled.csv, results/eager_vs_compiled.png

Part C - contiguous vs non-contiguous views (2048 x 2048 float32):
  the same operation on a contiguous tensor, its transpose (a view with swapped strides),
  a strided slice x[:, ::2] and a .contiguous() copy of the transpose.
  -> results/contiguity.csv, results/contiguity.png
"""
import platform
import time
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch._dynamo
import torch.fx
from torch import nn

from common import RESULTS, set_seeds, time_call

warnings.filterwarnings("ignore")
LINES = []


def log(text=""):
    print(text)
    LINES.append(str(text))


def describe(name, t):
    return (f"{name:<28} shape={tuple(t.shape)!s:<14} dtype={str(t.dtype):<14} "
            f"stride={t.stride()!s:<12} contiguous={t.is_contiguous()!s:<5} "
            f"storage_offset={t.storage_offset()}")


# ======================================================================= models
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Pointwise(nn.Module):
    """10 element-wise ops; in eager mode every op reads and writes the whole tensor."""

    def forward(self, x):
        y = x * 1.702
        y = torch.sigmoid(y)
        y = x * y                      # SiLU/GELU-like
        y = y + 0.5
        y = y * y
        y = torch.tanh(y)
        y = y - 0.1
        y = torch.relu(y)
        y = y * 3.0
        return y + x


def make_models():
    set_seeds()
    return {
        "tiny (2-8-1)": (nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()), 2),
        "mlp (784-512-512-10)": (nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512),
                                               nn.ReLU(), nn.Linear(512, 10)), 784),
        "pointwise (10 ops x 4096)": (Pointwise(), 4096),
    }


# ======================================================================= part A
def walk_autograd(fn, depth=0, seen=None):
    seen = set() if seen is None else seen
    if fn is None or fn in seen:
        return
    seen.add(fn)
    extra = ""
    if hasattr(fn, "variable"):
        v = fn.variable
        extra = f"  <- leaf tensor {tuple(v.shape)} requires_grad={v.requires_grad}"
    log("  " * depth + f"{type(fn).__name__}{extra}")
    for nxt, _ in fn.next_functions:
        walk_autograd(nxt, depth + 1, seen)


def part_a():
    log("=" * 90)
    log("PART A - graph inspection of SmallMLP (784 -> 128 -> 64 -> 10)")
    log("=" * 90)
    set_seeds()
    model = SmallMLP()
    x = torch.randn(32, 784)

    log("\n--- module tree")
    log(model)
    log(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    log("\n--- torch.fx symbolic trace (the forward pass as a static graph)")
    gm = torch.fx.symbolic_trace(model)
    log(gm.graph)
    log("generated code:")
    log(gm.code)

    log("\n--- autograd graph recorded by one forward pass (loss = out.sum())")
    out = model(x).sum()
    walk_autograd(out.grad_fn)

    log("\n--- TorchScript graph from torch.jit.trace")
    traced = torch.jit.trace(model.eval(), x)
    log(traced.graph)

    log("\n--- tensors: shape / dtype / stride / contiguity")
    log(describe("input x", x))
    for n, p in model.named_parameters():
        log(describe(n, p))
    w = model.fc1.weight
    log(describe("fc1.weight.t() (view)", w.t()))
    log(describe("x[:, ::2] (view)", x[:, ::2]))
    log(describe("x.t().contiguous() (copy)", x.t().contiguous()))
    log(describe("x.view(32, 28, 28)", x.view(32, 28, 28)))
    log(describe("x.view(32,28,28).permute(0,2,1)", x.view(32, 28, 28).permute(0, 2, 1)))
    log(f"x.t() shares storage with x: {x.t().untyped_storage().data_ptr() == x.untyped_storage().data_ptr()}")
    try:
        x.t().view(-1)
        log("x.t().view(-1): worked")
    except RuntimeError as e:
        log(f"x.t().view(-1) raises RuntimeError: {str(e).splitlines()[0]}")
    log(f"x.t().reshape(-1) works but copies: contiguous={x.t().reshape(-1).is_contiguous()}")
    log(f"nn.Linear computes x @ W.T, i.e. it reads fc1.weight {tuple(w.shape)} through a transposed view")


# ======================================================================= part B
def compile_modes(model, example):
    """Return {mode: (callable or None, first-call seconds, note)}."""
    modes = {"eager": (model, None, "")}

    def first_call(fn):
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(example)
        return time.perf_counter() - t0

    with torch.no_grad():
        try:
            tr = torch.jit.trace(model, example)
            modes["jit.trace"] = (tr, first_call(tr), "")
        except Exception as e:  # noqa: BLE001
            modes["jit.trace"] = (None, None, f"failed: {type(e).__name__}: {e}")
        try:
            fr = torch.jit.freeze(torch.jit.trace(model, example).eval())
            modes["jit.trace+freeze"] = (fr, first_call(fr), "")
        except Exception as e:  # noqa: BLE001
            modes["jit.trace+freeze"] = (None, None, f"failed: {type(e).__name__}: {e}")
    try:
        sc = torch.jit.script(model)
        modes["jit.script"] = (sc, first_call(sc), "")
    except Exception as e:  # noqa: BLE001
        modes["jit.script"] = (None, None, f"failed: {type(e).__name__}: {e}")
    try:
        torch._dynamo.reset()
        cm = torch.compile(model)
        modes["torch.compile"] = (cm, first_call(cm), "")
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        modes["torch.compile"] = (None, None, f"unsupported here: {type(e).__name__}: {msg}")
    return modes


def part_b():
    log("\n" + "=" * 90)
    log("PART B - eager vs compiled / traced")
    log("=" * 90)
    log(f"torch {torch.__version__}, {platform.system()} {platform.machine()}, "
        f"threads={torch.get_num_threads()}")
    rows = []
    for name, (model, d_in) in make_models().items():
        model.eval()
        for batch, reps in [(1, 200), (256, 50)]:
            set_seeds()
            x = torch.randn(batch, d_in)
            with torch.no_grad():
                ref = model(x)
            for mode, (fn, first, note) in compile_modes(model, x).items():
                if fn is None:
                    rows.append({"model": name, "batch": batch, "mode": mode, "note": note})
                    log(f"{name:<26} b={batch:<4} {mode:<17} {note}")
                    continue
                with torch.no_grad():
                    diff = (fn(x) - ref).abs().max().item()
                    med, _ = time_call(lambda: fn(x), warmup=20, reps=reps)
                rows.append({"model": name, "batch": batch, "mode": mode, "median_us": med * 1e6,
                             "first_call_ms": None if first is None else first * 1e3,
                             "max_abs_diff_vs_eager": diff, "note": note})
                log(f"{name:<26} b={batch:<4} {mode:<17} median {med * 1e6:9.1f} us"
                    + ("" if first is None else f"   first call {first * 1e3:8.1f} ms")
                    + f"   max|diff| {diff:.1e}")
    df = pd.DataFrame(rows)
    eager = df[df["mode"] == "eager"].set_index(["model", "batch"])["median_us"]
    df["speedup_vs_eager"] = [eager.get((m, b), np.nan) / t if pd.notna(t) else np.nan
                              for m, b, t in zip(df["model"], df["batch"], df.get("median_us"))]
    df.to_csv(RESULTS / "eager_vs_compiled.csv", index=False, float_format="%.4g")

    ok = df.dropna(subset=["median_us"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, batch in zip(axes, [1, 256]):
        sub = ok[ok["batch"] == batch].pivot(index="model", columns="mode", values="speedup_vs_eager")
        sub = sub[[c for c in ["eager", "jit.trace", "jit.trace+freeze", "jit.script", "torch.compile"]
                   if c in sub.columns]]
        sub.plot.bar(ax=ax, rot=0, width=0.8)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_title(f"batch {batch}")
        ax.set_ylabel("speed-up vs eager (×)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Eager vs compiled/traced inference (median after warm-up)")
    fig.tight_layout()
    fig.savefig(RESULTS / "eager_vs_compiled.png", dpi=130)


# ======================================================================= part C
def part_c(n=2048):
    log("\n" + "=" * 90)
    log(f"PART C - contiguous vs non-contiguous ({n} x {n} float32)")
    log("=" * 90)
    set_seeds()
    x = torch.randn(n, n)
    wide = torch.randn(n, 2 * n)
    variants = {
        "contiguous": x,
        "transposed view x.t()": x.t(),
        "strided slice [:, ::2]": wide[:, ::2],
        "x.t().contiguous() copy": x.t().contiguous(),
    }
    for k, v in variants.items():
        log(describe(k, v))
    w = torch.randn(n, n)
    out_full = torch.empty(n, n)
    out_row = torch.empty(n)
    # Outputs are written into pre-allocated buffers (out=) where PyTorch allows it, so the
    # timing measures the memory access pattern and not the allocation of a new 16 MB tensor.
    ops = {
        "elementwise add (out=)": (lambda v: torch.add(v, 1.0, out=out_full), 30),
        "row sum (out=)": (lambda v: torch.sum(v, dim=1, out=out_row), 30),
        "softmax(dim=1)": (lambda v: torch.softmax(v, dim=1), 30),
        "matmul v @ W (out=)": (lambda v: torch.matmul(v, w, out=out_full), 5),
        "copy into contiguous buffer": (lambda v: out_full.copy_(v), 30),
    }
    rows = []
    with torch.no_grad():
        for op, (fn, reps) in ops.items():
            for k, v in variants.items():
                med, _ = time_call(lambda: fn(v), warmup=5, reps=reps)
                out = fn(v)
                rows.append({"operation": op, "input": k, "input_contiguous": v.is_contiguous(),
                             "output_contiguous": out.is_contiguous(), "median_ms": med * 1e3})
                log(f"{op:<22} {k:<26} {med * 1e3:8.3f} ms   output contiguous={out.is_contiguous()}")
    df = pd.DataFrame(rows)
    base = df[df["input"] == "contiguous"].set_index("operation")["median_ms"]
    df["slowdown_vs_contiguous"] = df["median_ms"] / df["operation"].map(base)
    df.to_csv(RESULTS / "contiguity.csv", index=False, float_format="%.4g")

    piv = df.pivot(index="operation", columns="input", values="slowdown_vs_contiguous")[list(variants)]
    ax = piv.plot.bar(figsize=(10, 4.2), rot=0, width=0.8)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("time relative to contiguous input (×)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(f"Effect of memory layout on operation speed ({n}×{n} float32, CPU)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "contiguity.png", dpi=130)


def main():
    part_a()
    try:
        part_b()
    except Exception:  # keep part C even if a mode crashes in an unexpected way
        log(traceback.format_exc())
    part_c()
    (RESULTS / "graph_inspection.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
