"""Step 1: train the oversized baseline CNN (~1.0 M parameters) on Fashion-MNIST.

10 epochs, Adam lr 1e-3, batch 128. Records test accuracy, parameters, size on disk,
FLOPs, latency (batch 1 and 256) and the energy of training and of test-set inference.

Outputs: models/baseline.pt (not committed), results/baseline.csv, results/baseline_history.csv
"""
import pandas as pd
import torch

from common import (MODELS_DIR, RESULTS, Energy, count_params, evaluate, flops_per_image,
                    inference_energy, load_tensors, model_latency, serialized_bytes, set_seeds, train)
from models import BigCNN

EPOCHS = 10


def main():
    train_d, val_d, test_d = load_tensors()
    set_seeds()
    model = BigCNN()
    with Energy() as e:
        hist, t = train(model, train_d, val_d, EPOCHS, tag="baseline")
    torch.save(model.state_dict(), MODELS_DIR / "baseline.pt")

    kwh, kwh_1k = inference_energy(model, test_d[0])
    row = {"model": "baseline BigCNN", "params": count_params(model),
           "size_bytes": serialized_bytes(model.state_dict()),
           "size_on_disk_bytes": (MODELS_DIR / "baseline.pt").stat().st_size,
           "flops_per_image": flops_per_image(model),
           "val_acc": hist[-1]["val_acc"], "test_acc": evaluate(model, *test_d),
           "train_time_s": t, "train_energy_kwh": e.kwh,
           "inference_energy_kwh_per_1k": kwh_1k, **model_latency(model, test_d[0]),
           "threads": torch.get_num_threads()}
    pd.DataFrame([row]).to_csv(RESULTS / "baseline.csv", index=False, float_format="%.6g")
    pd.DataFrame(hist).to_csv(RESULTS / "baseline_history.csv", index=False, float_format="%.5f")
    for k, v in row.items():
        print(f"{k:<30} {v}")


if __name__ == "__main__":
    main()
