"""Step 2: accuracy against sparsity for unstructured (global, per layer) and structured pruning.

Output: results/sparsity_curves.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS


def main():
    un = pd.read_csv(RESULTS / "pruning_unstructured.csv")
    st = pd.read_csv(RESULTS / "pruning_structured.csv")
    base_acc = un[(un.method == "global") & (un.target_sparsity == 0)]["test_acc_after_pruning"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for method, color in [("global", "tab:blue"), ("layer", "tab:orange")]:
        d = un[un.method == method]
        ax.plot(d.target_sparsity, d.test_acc_after_pruning, "o--", color=color, alpha=0.6,
                label=f"{method}: directly after pruning")
        ax.plot(d.target_sparsity, d.test_acc_after_finetune, "o-", color=color,
                label=f"{method}: after 3-epoch fine-tune")
    ax.axhline(base_acc, color="k", lw=0.8, ls=":", label="baseline")
    ax.set_xlabel("weight sparsity")
    ax.set_ylabel("test accuracy")
    ax.set_title("Unstructured magnitude pruning")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    d = un[un.target_sparsity >= 0.5]
    for method, color in [("global", "tab:blue"), ("layer", "tab:orange")]:
        dd = d[d.method == method]
        ax.plot(dd.target_sparsity, dd.test_acc_after_finetune, "o-", color=color, label=method)
    ax.axhline(base_acc, color="k", lw=0.8, ls=":", label="baseline")
    ax.set_xlabel("weight sparsity")
    ax.set_title("Zoom: after fine-tuning, sparsity ≥ 0.5")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(st.param_sparsity_vs_baseline, st.test_acc_after_pruning, "s--", color="tab:green", alpha=0.6,
            label="structured: directly after pruning")
    ax.plot(st.param_sparsity_vs_baseline, st.test_acc_after_finetune, "s-", color="tab:green",
            label="structured: after fine-tune")
    for _, r in st.iterrows():
        ax.annotate(f"{r.filter_fraction_removed:.0%} filters", (r.param_sparsity_vs_baseline,
                    r.test_acc_after_finetune), textcoords="offset points", xytext=(4, 6), fontsize=7)
    g = un[un.method == "global"]
    ax.plot(g.target_sparsity, g.test_acc_after_finetune, "o:", color="tab:blue", alpha=0.7,
            label="unstructured global (fine-tuned)")
    ax.axhline(base_acc, color="k", lw=0.8, ls=":", label="baseline")
    ax.set_xlabel("fraction of parameters removed")
    ax.set_title("Structured (filter) vs unstructured pruning")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "sparsity_curves.png", dpi=130)


if __name__ == "__main__":
    main()
