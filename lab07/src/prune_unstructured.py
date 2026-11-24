"""Step 2a: unstructured magnitude pruning, global and per layer.

For every target sparsity s in [0, 0.25, 0.5, 0.75, 0.9, 0.95], starting each time from the
trained baseline (one-shot pruning):
  1. rank the weights of all Conv2d / Linear layers by |w| (biases are not pruned);
  2. global:    one threshold over all those weights together -> the s smallest |w| of the
                whole network become 0 (layers with many small weights lose more);
     per layer: every layer separately loses its own s smallest |w|;
  3. evaluate directly after pruning;
  4. fine-tune 3 epochs (Adam lr 3e-4) with the mask fixed (re-applied after every step)
     and evaluate again.

Outputs:
  results/pruning_unstructured.csv  - accuracy before/after fine-tuning per sparsity and method
  results/global_layer_sparsity.csv - how the global threshold distributes sparsity over layers
  models/unstructured_<method>_<s>.pt (not committed)
"""
import pandas as pd
import torch

from common import MODELS_DIR, RESULTS, Energy, count_nonzero, evaluate, load_tensors, set_seeds, train
from models import BigCNN, prunable_layers

SPARSITIES = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
FT_EPOCHS = 3
FT_LR = 3e-4


def kth_smallest_threshold(values, sparsity):
    """|w| threshold so that `sparsity` of `values` lie at or below it (k-th smallest)."""
    k = int(round(sparsity * values.numel()))
    if k == 0:
        return None
    return torch.kthvalue(values, k).values


def magnitude_masks(model, sparsity, scope):
    layers = prunable_layers(model)
    masks = {}
    if scope == "global":
        all_w = torch.cat([m.weight.detach().abs().flatten() for _, m in layers])
        thr = kth_smallest_threshold(all_w, sparsity)
        for _, m in layers:
            masks[m.weight] = (torch.ones_like(m.weight) if thr is None
                               else (m.weight.detach().abs() > thr).float())
    elif scope == "layer":
        for _, m in layers:
            thr = kth_smallest_threshold(m.weight.detach().abs().flatten(), sparsity)
            masks[m.weight] = (torch.ones_like(m.weight) if thr is None
                               else (m.weight.detach().abs() > thr).float())
    else:
        raise ValueError(scope)
    with torch.no_grad():
        for p, m in masks.items():
            p.mul_(m)
    return masks


def weight_sparsity(model):
    layers = prunable_layers(model)
    total = sum(m.weight.numel() for _, m in layers)
    zeros = sum(int((m.weight == 0).sum()) for _, m in layers)
    return zeros / total


def main():
    train_d, val_d, test_d = load_tensors()
    base_state = torch.load(MODELS_DIR / "baseline.pt", weights_only=True)
    rows, layer_rows = [], []
    for scope in ["global", "layer"]:
        for s in SPARSITIES:
            set_seeds()
            model = BigCNN()
            model.load_state_dict(base_state)
            masks = magnitude_masks(model, s, scope)
            acc_pruned = evaluate(model, *test_d)
            achieved = weight_sparsity(model)
            with Energy() as e:
                hist, t = train(model, train_d, val_d, FT_EPOCHS, lr=FT_LR, masks=masks,
                                tag=f"{scope} s={s}")
            acc_ft = evaluate(model, *test_d)
            assert abs(weight_sparsity(model) - achieved) < 1e-9, "mask was not kept during fine-tuning"
            rows.append({"method": scope, "target_sparsity": s, "achieved_weight_sparsity": achieved,
                         "nonzero_params": count_nonzero(model), "test_acc_after_pruning": acc_pruned,
                         "test_acc_after_finetune": acc_ft, "val_acc_after_finetune": hist[-1]["val_acc"],
                         "finetune_time_s": t, "finetune_energy_kwh": e.kwh})
            print(f"{scope:<6} s={s:<5} achieved={achieved:.4f}  acc pruned={acc_pruned:.4f}  "
                  f"fine-tuned={acc_ft:.4f}", flush=True)
            torch.save(model.state_dict(), MODELS_DIR / f"unstructured_{scope}_{s}.pt")
            if scope == "global":
                for name, m in prunable_layers(model):
                    layer_rows.append({"target_sparsity": s, "layer": name, "weights": m.weight.numel(),
                                       "layer_sparsity": float((m.weight == 0).float().mean())})
    pd.DataFrame(rows).to_csv(RESULTS / "pruning_unstructured.csv", index=False, float_format="%.6g")
    pd.DataFrame(layer_rows).to_csv(RESULTS / "global_layer_sparsity.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
