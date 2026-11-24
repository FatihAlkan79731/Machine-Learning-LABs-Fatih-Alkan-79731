"""Step 5: learning-rate schedules x optimizers under an equal 10-epoch budget.

Optimizers: SGD with momentum 0.9 (lr 0.05) and Adam (lr 1e-3), each with its usual
starting learning rate. Schedules: constant (reference), StepLR (x0.1 after 5 epochs),
CosineAnnealingLR (T_max = 10 epochs), ReduceLROnPlateau (x0.1 when the validation loss
does not improve for 1 epoch). Batch 128, 10 epochs for every run.

Outputs: results/schedule_study.csv, results/schedule_study.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS
from studies import history, run_train

OPTIMIZERS = {"sgd": ["optimizer.name=sgd", "optimizer.lr=0.05", "optimizer.momentum=0.9"],
              "adam": ["optimizer.name=adam", "optimizer.lr=0.001"]}
SCHEDULES = {"none": ["scheduler.name=none"],
             "step": ["scheduler.name=step", "scheduler.step_size=5", "scheduler.gamma=0.1"],
             "cosine": ["scheduler.name=cosine", "scheduler.t_max=10"],
             "plateau": ["scheduler.name=plateau", "scheduler.patience=1", "scheduler.factor=0.1"]}


def main():
    rows = []
    for opt, o_args in OPTIMIZERS.items():
        for sch, s_args in SCHEDULES.items():
            name = f"sched_{opt}_{sch}"
            s = run_train(name, o_args + s_args + ["training.epochs=10"], tags={"step": "5-schedules"})
            h = history(name)
            rows.append({"optimizer": opt, "scheduler": sch, "final_val_acc": s["final_val_acc"],
                         "best_val_acc": s["best_val_acc"], "test_acc": s["test_acc"],
                         "final_train_loss": h["train_loss"].iloc[-1],
                         "final_val_loss": h["val_loss"].iloc[-1],
                         "final_lr": h["lr"].iloc[-1], "train_time_s": s["train_time_s"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "schedule_study.csv", index=False, float_format="%.5g")
    print("\n" + df.round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for r, opt in enumerate(OPTIMIZERS):
        for sch in SCHEDULES:
            h = history(f"sched_{opt}_{sch}")
            axes[r, 0].plot(h["epoch"], h["train_loss"], marker=".", label=sch)
            axes[r, 1].plot(h["epoch"], h["val_acc"], marker=".", label=sch)
            axes[r, 2].plot(h["epoch"], h["lr"], marker=".", label=sch)
        axes[r, 0].set_ylabel(f"{opt.upper()}\ntrain loss")
        axes[r, 1].set_ylabel("validation accuracy")
        axes[r, 2].set_ylabel("learning rate (start of epoch)")
        axes[r, 2].set_yscale("log")
        for ax in axes[r]:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("LR schedules × optimizers, 10 epochs each")
    fig.tight_layout()
    fig.savefig(RESULTS / "schedule_study.png", dpi=130)


if __name__ == "__main__":
    main()
