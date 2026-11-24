"""Step 7: interrupt training at epoch 3, resume from the checkpoint, compare with an
uninterrupted run.

Setting: 6 epochs, Adam lr 1e-3 with CosineAnnealingLR (T_max 6), so the scheduler state
must also be restored correctly for the learning rate to continue on the same curve.
  * resume_reference   - 6 epochs without interruption
  * resume_interrupted - stopped after epoch 3 (train.py --stop-after 3), then a new process
                         resumes from checkpoints/resume_interrupted/epoch_03.pt and finishes
                         epochs 4-6 in the same MLflow run

Outputs: results/resume_study.csv, results/resume_study.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS, ROOT, load_config
from studies import history, run_train

ARGS = ["training.epochs=6", "scheduler.name=cosine", "scheduler.t_max=6"]


def main():
    cfg = load_config()
    ref = run_train("resume_reference", ARGS, tags={"step": "7-resume"})
    run_train("resume_interrupted", ARGS, extra=["--stop-after", "3"], tags={"step": "7-resume"})
    ckpt = ROOT / cfg.checkpoint.dir / "resume_interrupted" / "epoch_03.pt"
    res = run_train("resume_interrupted", extra=["--resume", str(ckpt)])

    h_ref, h_res = history("resume_reference"), history("resume_interrupted")
    df = h_ref.merge(h_res, on="epoch", suffixes=("_reference", "_resumed"))
    for c in ["lr", "train_loss", "val_loss", "val_acc"]:
        df[f"abs_diff_{c}"] = (df[f"{c}_reference"] - df[f"{c}_resumed"]).abs()
    keep = ["epoch"] + [f"{c}_{s}" for c in ["lr", "train_loss", "val_acc"] for s in ["reference", "resumed"]] \
        + [c for c in df.columns if c.startswith("abs_diff")]
    df[keep].to_csv(RESULTS / "resume_study.csv", index=False, float_format="%.8g")
    print(df[keep].to_string(index=False))
    print(f"test accuracy: reference {ref['test_acc']:.4f}, interrupted+resumed {res['test_acc']:.4f}; "
          f"same MLflow run id before/after resume: {res['mlflow_run_id']}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col in zip(axes, ["train_loss", "val_acc", "lr"]):
        ax.plot(h_ref["epoch"], h_ref[col], "o-", lw=3, alpha=0.5, label="uninterrupted")
        ax.plot(h_res["epoch"][:3], h_res[col][:3], "x-", color="tab:red", label="run 1 (epochs 1-3)")
        ax.plot(h_res["epoch"][3:], h_res[col][3:], "x--", color="tab:green", label="resumed (epochs 4-6)")
        ax.axvline(3.5, color="gray", ls=":")
        ax.set_xlabel("epoch")
        ax.set_title(col)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Interrupt after epoch 3 and resume from checkpoint")
    fig.tight_layout()
    fig.savefig(RESULTS / "resume_study.png", dpi=130)


if __name__ == "__main__":
    main()
