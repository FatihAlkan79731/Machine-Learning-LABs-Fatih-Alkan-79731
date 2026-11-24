"""Step 5: one table organised by the Ch. 9 efficiency pillars (data, model, compute, energy).

Models: dense baseline, global unstructured pruning 90 % (fine-tuned), structured pruning
50 % of filters (fine-tuned), plain student and distilled student (manual loss, seed 42).
Latency and inference energy are measured again here, in one process, so all rows are
comparable.

Outputs: results/efficiency_summary.csv, results/efficiency_summary.md
"""
import numbers

import pandas as pd
import torch

from common import (MODELS_DIR, RESULTS, count_nonzero, count_params, evaluate, flops_per_image,
                    inference_energy, load_tensors, model_latency, serialized_bytes)
from latency_size import bitmask_state, sparse_state
from models import BigCNN, SmallCNN, prunable_layers

N_TRAIN = 55_000


def load(kind, path, channels=None):
    m = BigCNN(channels=channels) if kind == "big" else SmallCNN()
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()


def main():
    _, _, (xte, yte) = load_tensors()
    base_csv = pd.read_csv(RESULTS / "baseline.csv").iloc[0]
    un = pd.read_csv(RESULTS / "pruning_unstructured.csv")
    st = pd.read_csv(RESULTS / "pruning_structured.csv")
    runs = pd.read_csv(RESULTS / "distillation_runs.csv")
    g90 = un[(un.method == "global") & (un.target_sparsity == 0.9)].iloc[0]
    s50 = st[st.filter_fraction_removed == 0.5].iloc[0]
    sd = torch.load(MODELS_DIR / "structured_0.5.pt", weights_only=True)
    ch50 = tuple(sd[f"conv{i}.weight"].shape[0] for i in (1, 2, 3))

    base_train_kwh = base_csv["train_energy_kwh"]
    models = [
        ("Dense baseline", load("big", MODELS_DIR / "baseline.pt"),
         "10 epochs, hard labels", 10 * N_TRAIN, base_train_kwh, "-"),
        ("Unstructured global 90 %", load("big", MODELS_DIR / "unstructured_global_0.9.pt"),
         "baseline + 3 fine-tune epochs", 13 * N_TRAIN, base_train_kwh + g90.finetune_energy_kwh,
         "needs the trained baseline"),
        ("Structured 50 % filters", load("big", MODELS_DIR / "structured_0.5.pt", ch50),
         "baseline + 3 fine-tune epochs", 13 * N_TRAIN, base_train_kwh + s50.finetune_energy_kwh,
         "needs the trained baseline"),
        ("Student, no teacher", load("small", MODELS_DIR / "student_plain_42.pt"),
         "10 epochs, hard labels", 10 * N_TRAIN,
         runs[(runs.variant == "plain") & (runs.seed == 42)].train_energy_kwh.iloc[0], "-"),
        ("Student, distilled (T=4, α=0.5)", load("small", MODELS_DIR / "student_kd_manual_42.pt"),
         "10 epochs, hard + teacher soft labels", 10 * N_TRAIN + N_TRAIN,
         base_train_kwh + runs[(runs.variant == "kd_manual") & (runs.seed == 42)].train_energy_kwh.iloc[0],
         "needs the trained teacher + 1 teacher pass"),
    ]
    rows = []
    for name, m, data_recipe, samples, train_kwh, dep in models:
        density = {mod: float((mod.weight != 0).float().mean()) for _, mod in prunable_layers(m)}
        sparse_bytes = min(serialized_bytes(sparse_state(m, torch.int32)), serialized_bytes(bitmask_state(m)))
        lat = model_latency(m, xte)
        _, kwh_1k = inference_energy(m, xte)
        rows.append({
            "model": name,
            # data
            "data_training_recipe": data_recipe, "data_samples_processed": samples, "data_depends_on": dep,
            # model
            "model_params": count_params(m), "model_nonzero_params": count_nonzero(m),
            "model_size_dense_MB": serialized_bytes(m.state_dict()) / 2 ** 20,
            "model_size_best_sparse_MB": sparse_bytes / 2 ** 20,
            "test_acc": evaluate(m, xte, yte),
            # compute
            "compute_flops_executed": flops_per_image(m),
            "compute_flops_nonzero": flops_per_image(m, density),
            "compute_latency_b1_ms": lat["latency_b1_ms"], "compute_latency_b256_ms": lat["latency_b256_ms"],
            # energy
            "energy_training_kwh_total": train_kwh,
            "energy_inference_kwh_per_1k_images": kwh_1k,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "efficiency_summary.csv", index=False, float_format="%.6g")

    # transposed markdown table grouped by pillar, for the report
    pillar = {"data": "Data", "model": "Model", "test": "Model", "compute": "Compute", "energy": "Energy"}
    lines = ["| Pillar | Metric | " + " | ".join(df.model) + " |",
             "|---|---|" + "---|" * len(df)]
    for col in df.columns[1:]:
        p = pillar[col.split("_")[0]]
        vals = []
        for v in df[col]:
            if isinstance(v, numbers.Integral):
                vals.append(f"{v:,}")
            elif isinstance(v, numbers.Real):
                vals.append(f"{v:.3g}" if v != v or abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append(f"| {p} | {col} | " + " | ".join(vals) + " |")
    (RESULTS / "efficiency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
