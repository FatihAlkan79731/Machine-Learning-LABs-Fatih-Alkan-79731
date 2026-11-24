"""Step 3: latency and on-disk size of dense vs sparse models.

A) Size of the globally pruned models in five storage formats:
   dense float32 state_dict; dense + gzip; CSR with int64 indices (PyTorch default);
   CSR with int32 indices; values + 1-bit mask (lower bound for index overhead).
B) Latency (batch 1 and 256, median after warm-up) of
   the dense baseline; the 90 % / 95 % globally pruned models stored dense (zeros are
   still multiplied); the same models with fc1/fc2 executed as sparse CSR matmuls
   (convolutions stay dense - PyTorch has no sparse CPU convolution); and the
   structurally pruned (physically smaller) models.
C) Micro-benchmark of fc1 alone (144 x 6272): dense matmul vs CSR matmul for sparsity
   0 ... 0.99, batch 1 and 256 - where does sparse start to pay off?

Outputs: results/size_formats.csv, results/latency_sparse.csv, results/sparse_matmul_micro.csv,
         results/latency_size.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from common import (MODELS_DIR, RESULTS, evaluate, flops_per_image, latency_ms, load_tensors,
                    serialized_bytes, set_seeds)
from models import BigCNN, prunable_layers

SPARSITIES = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
STRUCT = [0.10, 0.25, 0.50]


# ------------------------------------------------------------------ storage formats
def sparse_state(model, index_dtype):
    out = {}
    for name, t in model.state_dict().items():
        if name.endswith("weight") and t.dim() > 1:
            csr = t.reshape(t.shape[0], -1).to_sparse_csr()
            out[name] = {"shape": tuple(t.shape), "values": csr.values().clone(),
                         "crow": csr.crow_indices().to(index_dtype).clone(),
                         "col": csr.col_indices().to(index_dtype).clone()}
        else:
            out[name] = t.clone()
    return out


def bitmask_state(model):
    out = {}
    for name, t in model.state_dict().items():
        if name.endswith("weight") and t.dim() > 1:
            mask = (t != 0).flatten().numpy()
            out[name] = {"shape": tuple(t.shape), "values": t.flatten()[t.flatten() != 0].clone(),
                         "mask": torch.from_numpy(np.packbits(mask))}
        else:
            out[name] = t.clone()
    return out


def size_row(model, label, sparsity):
    sd = model.state_dict()
    return {"model": label, "weight_sparsity": sparsity,
            "dense_bytes": serialized_bytes(sd),
            "dense_gzip_bytes": serialized_bytes(sd, compress=True),
            "csr_int64_bytes": serialized_bytes(sparse_state(model, torch.int64)),
            "csr_int32_bytes": serialized_bytes(sparse_state(model, torch.int32)),
            "values_plus_bitmask_bytes": serialized_bytes(bitmask_state(model))}


# ------------------------------------------------------------------ sparse execution
def sparse_weight(w):
    """CSR copy of w if this PyTorch build can multiply CSR x dense on the CPU, else COO."""
    try:
        csr = w.to_sparse_csr()
        torch.sparse.mm(csr, torch.ones(w.shape[1], 1))
        return csr
    except (RuntimeError, NotImplementedError):
        print("  CSR matmul not available, using COO")
        return w.to_sparse_coo()


class SparseLinear(nn.Module):
    """y = x W^T + b with W stored as a sparse (CSR) matrix."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.w = sparse_weight(lin.weight.detach())
        self.b = lin.bias.detach().clone()

    def forward(self, x):
        return torch.sparse.mm(self.w, x.t()).t() + self.b


def with_sparse_linears(model):
    model.fc1 = SparseLinear(model.fc1)
    model.fc2 = SparseLinear(model.fc2)
    return model.eval()


def load_big(path, channels=(32, 64, 128)):
    m = BigCNN(channels=channels)
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()


def lat(model, x_test):
    return {"latency_b1_ms": latency_ms(model, x_test[:1], 20, 200),
            "latency_b256_ms": latency_ms(model, x_test[:256], 5, 30)}


def main():
    _, _, (x_test, y_test) = load_tensors()

    # ---------------------------------------------------------------- A) size
    size_rows = []
    for s in SPARSITIES:
        size_rows.append(size_row(load_big(MODELS_DIR / f"unstructured_global_{s}.pt"), f"global {s:.0%}", s))
    for f in STRUCT:
        sd = torch.load(MODELS_DIR / f"structured_{f}.pt", weights_only=True)
        ch = (sd["conv1.weight"].shape[0], sd["conv2.weight"].shape[0], sd["conv3.weight"].shape[0])
        size_rows.append(size_row(load_big(MODELS_DIR / f"structured_{f}.pt", ch), f"structured {f:.0%} filters", 0))
    sizes = pd.DataFrame(size_rows)
    for c in [c for c in sizes.columns if c.endswith("_bytes")]:
        sizes[c.replace("_bytes", "_MB")] = sizes[c] / 2 ** 20
    sizes.to_csv(RESULTS / "size_formats.csv", index=False, float_format="%.4f")
    print(sizes[["model"] + [c for c in sizes.columns if c.endswith("_MB")]].round(3).to_string(index=False))

    # ---------------------------------------------------------------- B) latency
    lat_rows = []
    base = load_big(MODELS_DIR / "baseline.pt")
    lat_rows.append({"model": "baseline (dense)", "format": "dense", **lat(base, x_test),
                     "test_acc": evaluate(base, x_test, y_test), "flops_executed": flops_per_image(base)})
    for s in [0.9, 0.95]:
        m = load_big(MODELS_DIR / f"unstructured_global_{s}.pt")
        density = {mod: float((mod.weight != 0).float().mean()) for _, mod in prunable_layers(m)}
        useful = flops_per_image(m, density)
        acc = evaluate(m, x_test, y_test)
        lat_rows.append({"model": f"global {s:.0%} sparse", "format": "dense storage (zeros kept)",
                         **lat(m, x_test), "test_acc": acc, "flops_executed": flops_per_image(m),
                         "flops_nonzero": useful})
        with torch.no_grad():
            dense_out = m(x_test[:256])
            ms = with_sparse_linears(m)
            diff = (ms(x_test[:256]) - dense_out).abs().max().item()
        lat_rows.append({"model": f"global {s:.0%} sparse", "format": "fc layers as CSR",
                         **lat(ms, x_test), "test_acc": evaluate(ms, x_test, y_test),
                         "flops_nonzero": useful, "max_abs_diff_vs_dense_format": diff})
    for f in STRUCT:
        sd = torch.load(MODELS_DIR / f"structured_{f}.pt", weights_only=True)
        ch = (sd["conv1.weight"].shape[0], sd["conv2.weight"].shape[0], sd["conv3.weight"].shape[0])
        m = load_big(MODELS_DIR / f"structured_{f}.pt", ch)
        lat_rows.append({"model": f"structured {f:.0%} filters", "format": "smaller dense",
                         **lat(m, x_test), "test_acc": evaluate(m, x_test, y_test),
                         "flops_executed": flops_per_image(m)})
    lats = pd.DataFrame(lat_rows)
    b1, b256 = lats.loc[0, "latency_b1_ms"], lats.loc[0, "latency_b256_ms"]
    lats["speedup_b1"] = b1 / lats["latency_b1_ms"]
    lats["speedup_b256"] = b256 / lats["latency_b256_ms"]
    lats.to_csv(RESULTS / "latency_sparse.csv", index=False, float_format="%.5g")
    print("\n" + lats.round(4).to_string(index=False))

    # ---------------------------------------------------------------- C) micro-benchmark
    set_seeds()
    w_full = torch.randn(144, 6272)
    micro = []
    for s in [0, 0.5, 0.75, 0.9, 0.95, 0.99]:
        w = w_full.clone()
        if s > 0:
            thr = torch.kthvalue(w.abs().flatten(), int(s * w.numel())).values
            w[w.abs() <= thr] = 0
        w_csr = sparse_weight(w)
        for batch in [1, 256]:
            x = torch.randn(batch, 6272)
            xt = x.t().contiguous()
            d = latency_ms(lambda v: v @ w.t(), x, 20, 200)
            sp = latency_ms(lambda v: torch.sparse.mm(w_csr, v), xt, 20, 200)
            micro.append({"sparsity": s, "batch": batch, "dense_ms": d, "csr_ms": sp,
                          "csr_speedup": d / sp, "nnz": int((w != 0).sum()), "layout": str(w_csr.layout)})
            print(f"fc1 micro: sparsity {s:<5} batch {batch:<4} dense {d:.4f} ms  CSR {sp:.4f} ms  "
                  f"speed-up {d / sp:.2f}x", flush=True)
    micro = pd.DataFrame(micro)
    micro.to_csv(RESULTS / "sparse_matmul_micro.csv", index=False, float_format="%.5g")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    g = sizes[sizes.model.str.startswith("global")]
    for c, lbl in [("dense_MB", "dense float32"), ("dense_gzip_MB", "dense + gzip"),
                   ("csr_int64_MB", "CSR, int64 indices"), ("csr_int32_MB", "CSR, int32 indices"),
                   ("values_plus_bitmask_MB", "values + bitmask")]:
        ax.plot(g.weight_sparsity, g[c], "o-", label=lbl)
    ax.set_xlabel("weight sparsity (global pruning)")
    ax.set_ylabel("serialized size (MB)")
    ax.set_title("On-disk size: dense vs sparse formats")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [f"{m}\n{f}" for m, f in zip(lats.model, lats.format)]
    y = np.arange(len(lats))
    ax.barh(y - 0.2, lats.speedup_b1, height=0.4, label="batch 1")
    ax.barh(y + 0.2, lats.speedup_b256, height=0.4, label="batch 256")
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("speed-up vs dense baseline (×)")
    ax.set_title("Real latency")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    for batch, st in [(1, "o-"), (256, "s--")]:
        d = micro[micro.batch == batch]
        ax.plot(d.sparsity, d.csr_speedup, st, label=f"batch {batch}")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("weight sparsity of fc1 (144 × 6272)")
    ax.set_ylabel("CSR speed-up over dense matmul (×)")
    ax.set_yscale("log")
    ax.set_title("When does sparse matmul pay off?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "latency_size.png", dpi=130)


if __name__ == "__main__":
    main()
