"""Step 4: compare the measured cost of each model with the deployment budgets.

A model fits a target if all three checks pass:
  memory  - peak RSS during inference is within the target's memory range
  latency - median single-sample latency is within the limit
  size    - serialized model size is within the limit
For the minimum of a memory range (Cloud >= 1 GB, Edge >= 256 MB, ...) we only
need the device to have at least that much, so only the upper bound is checked
against the model. Cloud has no upper bound.
"""
import pandas as pd

from common import RESULTS

KB = 1024
MB = 1024 * 1024

# max memory (bytes, None = no limit), max latency (ms), max size (bytes)
BUDGETS = {
    "Cloud":  {"memory": "≥ 1 GB",        "max_mem": None,     "max_lat": 100, "max_size": 500 * MB},
    "Edge":   {"memory": "256–1024 MB",   "max_mem": 1024 * MB, "max_lat": 50,  "max_size": 50 * MB},
    "Mobile": {"memory": "64–256 MB",     "max_mem": 256 * MB, "max_lat": 20,  "max_size": 10 * MB},
    "TinyML": {"memory": "≤ 256 KB",      "max_mem": 256 * KB, "max_lat": 10,  "max_size": 100 * KB},
}


def ok(flag):
    return "yes" if flag else "no"


def main():
    cost = pd.read_csv(RESULTS / "system_cost.csv")

    rows = []
    for _, m in cost.iterrows():
        mem = m["peak_rss_infer_mb"] * MB
        lat = m["infer_latency_ms"]
        size = m["model_size_bytes"]
        for target, b in BUDGETS.items():
            mem_ok = b["max_mem"] is None or mem <= b["max_mem"]
            lat_ok = lat <= b["max_lat"]
            size_ok = size <= b["max_size"]
            rows.append({
                "model": m["model"],
                "target": target,
                "memory_budget": b["memory"],
                "latency_budget_ms": b["max_lat"],
                "size_budget_kb": b["max_size"] / KB,
                "memory_ok": ok(mem_ok),
                "latency_ok": ok(lat_ok),
                "size_ok": ok(size_ok),
                "fits": ok(mem_ok and lat_ok and size_ok),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "budget_fit.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
