"""Step 3: parameters, FLOPs (by hand and with thop) and inference latency.

Hand FLOPs count only the multiply-adds of linear / conv / attention products
(1 multiply-add = 2 FLOPs). Bias additions, activations, softmax, pooling and
LayerNorm are left out, as in the Ch. 4 formulas. thop reports MACs, so its
numbers are multiplied by 2 before comparing.

Outputs:
  results/model_summaries.txt  - torchinfo summaries
  results/flops_layer.csv      - one layer of each type: hand vs thop
  results/flops_model.csv      - whole model: params, hand vs thop FLOPs
  results/latency.csv          - median latency for batch 1 (100 runs) and 128 (20 runs)
"""
import copy
import statistics
import time
import warnings

import pandas as pd
import thop
import torch
from torch import nn
from torchinfo import summary

from common import RESULTS, set_seeds
from models import MODELS, build

warnings.filterwarnings("ignore")
SEQ, D, FF, HEADS = 16, 64, 128, 4


# ---------------------------------------------------------------- hand formulas
def linear_flops(n_in, n_out, seq=1):
    return 2 * seq * n_in * n_out


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def encoder_layer_flops(seq=SEQ, d=D, ff=FF):
    parts = {
        "qkv_projection": 2 * seq * d * 3 * d,
        "attention_scores (QK^T)": 2 * seq * seq * d,
        "weighted_sum (AV)": 2 * seq * seq * d,
        "output_projection": 2 * seq * d * d,
        "feed_forward (2 linears)": 2 * seq * d * ff * 2,
    }
    return sum(parts.values()), parts


def hand_model_flops(name):
    if name == "MLP":
        return linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10)
    if name == "CNN":
        return (conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14)
                + linear_flops(32 * 7 * 7, 10))
    if name == "Attention":
        return (linear_flops(49, D, seq=SEQ) + 2 * encoder_layer_flops()[0]
                + linear_flops(D, 10))
    raise ValueError(name)


# ---------------------------------------------------------------- thop
def thop_flops(module, x):
    """thop MACs x 2. The module is copied and put in train mode so that the
    TransformerEncoderLayer does not take its fused inference fast path, which
    would hide the inner Linear layers from thop's hooks."""
    m = copy.deepcopy(module).train()
    with torch.no_grad():
        macs, _ = thop.profile(m, inputs=(x,), verbose=False)
    return 2 * int(macs)


# ---------------------------------------------------------------- latency
def median_latency_ms(model, batch, runs, warmup):
    x = torch.randn(batch, 1, 28, 28)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def main():
    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    x1 = torch.randn(1, 1, 28, 28)

    # parameters + torchinfo
    summaries, model_rows, lat_rows = [], [], []
    for name in MODELS:
        set_seeds()
        model = build(name)
        try:
            info = summary(model, input_size=(1, 1, 28, 28), verbose=0, mode="train",
                           col_names=("output_size", "num_params"), depth=4)
        except TypeError:
            info = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                           col_names=("output_size", "num_params"), depth=4)
        summaries.append(f"===== {name} =====\n{info}\n")
        n_params = sum(p.numel() for p in model.parameters())

        hand = hand_model_flops(name)
        measured = thop_flops(model, x1)
        model_rows.append({
            "model": name,
            "params_torchinfo": info.total_params,
            "params_counted": n_params,
            "hand_flops": hand,
            "thop_flops": measured,
            "thop_over_hand": measured / hand,
        })

        lat_rows.append({
            "model": name,
            "latency_b1_ms": median_latency_ms(model, 1, runs=100, warmup=10),
            "latency_b128_ms": median_latency_ms(model, 128, runs=20, warmup=3),
        })
        print(f"{name}: {n_params:,} params, hand {hand:,} FLOPs, thop {measured:,} FLOPs")

    (RESULTS / "model_summaries.txt").write_text("\n".join(summaries))

    # one layer of each type
    total, parts = encoder_layer_flops()
    enc_layer = build("Attention").encoder.layers[0]
    layer_rows = [
        {"model": "MLP", "layer": "Linear(784, 256)",
         "formula": "2 x in x out",
         "hand_flops": linear_flops(784, 256),
         "thop_flops": thop_flops(nn.Linear(784, 256), torch.randn(1, 784))},
        {"model": "CNN", "layer": "Conv2d(16, 32, 3) on 14x14",
         "formula": "2 x K x K x C_in x C_out x H_out x W_out",
         "hand_flops": conv_flops(3, 16, 32, 14, 14),
         "thop_flops": thop_flops(nn.Conv2d(16, 32, 3, padding=1), torch.randn(1, 16, 14, 14))},
        {"model": "Attention", "layer": "TransformerEncoderLayer (seq 16, d 64, ff 128)",
         "formula": " + ".join(f"{k} {v:,}" for k, v in parts.items()),
         "hand_flops": total,
         "thop_flops": thop_flops(enc_layer, torch.randn(1, SEQ, D))},
    ]
    for r in layer_rows:
        r["thop_over_hand"] = r["thop_flops"] / r["hand_flops"]

    pd.DataFrame(layer_rows).to_csv(RESULTS / "flops_layer.csv", index=False, float_format="%.4f")
    pd.DataFrame(model_rows).to_csv(RESULTS / "flops_model.csv", index=False, float_format="%.4f")
    lat = pd.DataFrame(lat_rows)
    lat["torch_threads"] = torch.get_num_threads()
    lat.to_csv(RESULTS / "latency.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print("\n" + pd.DataFrame(layer_rows).drop(columns="formula").to_string(index=False))
        print("\n" + pd.DataFrame(model_rows).to_string(index=False))
        print("\n" + lat.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
