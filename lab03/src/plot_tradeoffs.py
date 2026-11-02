"""Step 5: test accuracy against parameter count and against latency."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS

COLORS = {"MLP": "#1f77b4", "CNN": "#ff7f0e", "Attention": "#2ca02c"}


def main():
    acc = pd.read_csv(RESULTS / "test_accuracy.csv")
    cost = pd.read_csv(RESULTS / "flops_model.csv")
    lat = pd.read_csv(RESULTS / "latency.csv")
    df = acc.merge(cost, on="model").merge(lat, on="model")
    df.to_csv(RESULTS / "tradeoff_table.csv", index=False, float_format="%.4f")

    panels = [
        ("params_counted", "parameters", False),
        ("latency_b1_ms", "latency, batch 1 (ms, median of 100)", False),
        ("latency_b128_ms", "latency, batch 128 (ms, median of 20)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (col, label, logx) in zip(axes, panels):
        for _, r in df.iterrows():
            ax.scatter(r[col], r["test_acc"] * 100, s=80, color=COLORS[r["model"]], zorder=3)
            ax.annotate(r["model"], (r[col], r["test_acc"] * 100),
                        textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlim(0, df[col].max() * 1.25)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Fashion-MNIST, 5 epochs: accuracy vs. cost")
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff_plots.png", dpi=150)
    print(df[["model", "test_acc", "params_counted", "hand_flops", "latency_b1_ms",
              "latency_b128_ms"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
