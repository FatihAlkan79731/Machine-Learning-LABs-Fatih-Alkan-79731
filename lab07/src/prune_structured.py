"""Step 2b: structured pruning - remove whole conv filters by L2 norm (10 %, 25 %, 50 %).

In each of conv1, conv2 and conv3 the given fraction of output filters with the smallest
L2 norm is removed, together with the matching input channels of the next layer (and the
49 matching input features of fc1 after conv3). The result is rebuilt as a smaller *dense*
BigCNN, so the savings in parameters, FLOPs and latency are real. Then 3 epochs of
fine-tuning (Adam lr 3e-4), as for unstructured pruning.

Outputs: results/pruning_structured.csv, models/structured_<f>.pt (not committed)
"""
import pandas as pd
import torch

from common import (MODELS_DIR, RESULTS, Energy, count_params, evaluate, flops_per_image, load_tensors,
                    model_latency, serialized_bytes, set_seeds, train)
from models import BigCNN, structured_prune

FRACTIONS = [0.0, 0.10, 0.25, 0.50]
FT_EPOCHS = 3
FT_LR = 3e-4


def main():
    train_d, val_d, test_d = load_tensors()
    base = BigCNN()
    base.load_state_dict(torch.load(MODELS_DIR / "baseline.pt", weights_only=True))
    base_params = count_params(base)
    rows = []
    for f in FRACTIONS:
        set_seeds()
        small, channels = structured_prune(base, f)
        acc_pruned = evaluate(small, *test_d)
        if f > 0:
            with Energy() as e:
                hist, t = train(small, train_d, val_d, FT_EPOCHS, lr=FT_LR, tag=f"structured {f:.0%}")
            kwh = e.kwh
            acc_ft = evaluate(small, *test_d)
        else:   # 0 % = the baseline itself; its fine-tuned version is the unstructured s=0 run
            t, acc_ft, kwh = 0.0, acc_pruned, 0.0
        torch.save(small.state_dict(), MODELS_DIR / f"structured_{f}.pt")
        p = count_params(small)
        rows.append({"filter_fraction_removed": f, "channels": "-".join(map(str, channels)),
                     "params": p, "param_sparsity_vs_baseline": 1 - p / base_params,
                     "size_bytes": serialized_bytes(small.state_dict()),
                     "flops_per_image": flops_per_image(small),
                     "test_acc_after_pruning": acc_pruned, "test_acc_after_finetune": acc_ft,
                     "finetune_time_s": t, "finetune_energy_kwh": kwh, **model_latency(small, test_d[0])})
        print(f"structured {f:.0%}: channels {channels}, params {p:,}, acc pruned {acc_pruned:.4f}, "
              f"fine-tuned {acc_ft:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pruning_structured.csv", index=False, float_format="%.5g")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
