"""Step 4: knowledge distillation into a ~100k-parameter student.

Teacher  = the oversized baseline (models/baseline.pt), frozen, eval mode.
Student  = SmallCNN, 103,750 parameters (about 10x smaller than the teacher).
Loss (as in the lab manual), T = 4, alpha = 0.5:
    L = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL( softmax(s/T) || softmax(t/T) )
The manual writes the KL term with the student distribution first, KL(p_s || p_t). The
original formulation of Hinton et al. (2015) uses KL(p_t || p_s) (the teacher is the
target, as in torch.nn.KLDivLoss(log_softmax(s/T), softmax(t/T))). The manual's form is
the main experiment; the other direction is run as an extra comparison.

Runs (10 epochs, Adam lr 1e-3, batch 128; same initial student weights per seed):
  plain            - student trained on labels only (CE)
  kd_manual        - the loss above, KL(p_s || p_t)
  kd_hinton        - same, with KL(p_t || p_s)
Each variant is trained with 3 seeds (42, 43, 44) because the expected differences are
small; mean and standard deviation are reported.

Outputs: results/distillation_runs.csv, results/distillation.csv,
         results/distillation_history.csv, results/distillation.png,
         models/student_<variant>_<seed>.pt (not committed)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from common import (MODELS_DIR, RESULTS, Energy, count_params, evaluate, flops_per_image,
                    inference_energy, load_tensors, model_latency, predict_logits, serialized_bytes,
                    set_seeds, train)
from models import BigCNN, SmallCNN

T = 4.0
ALPHA = 0.5
EPOCHS = 10
SEEDS = [42, 43, 44]


def kd_loss(direction):
    def loss(s_logits, y, t_logits):
        ce = F.cross_entropy(s_logits, y)
        log_ps = F.log_softmax(s_logits / T, dim=1)
        log_pt = F.log_softmax(t_logits / T, dim=1)
        if direction == "manual":        # KL(p_s || p_t) = sum p_s (log p_s - log p_t)
            kl = (log_ps.exp() * (log_ps - log_pt)).sum(1).mean()
        else:                            # KL(p_t || p_s) = sum p_t (log p_t - log p_s)
            kl = (log_pt.exp() * (log_pt - log_ps)).sum(1).mean()
        return ALPHA * ce + (1 - ALPHA) * T * T * kl
    return loss


def plain_loss(s_logits, y, _t_logits):
    return F.cross_entropy(s_logits, y)


VARIANTS = {"plain": plain_loss, "kd_manual": kd_loss("manual"), "kd_hinton": kd_loss("hinton")}


def main():
    (xtr, ytr), val_d, (xte, yte) = load_tensors()
    teacher = BigCNN()
    teacher.load_state_dict(torch.load(MODELS_DIR / "baseline.pt", weights_only=True))
    teacher.eval()
    t_train = predict_logits(teacher, xtr)          # soft targets, computed once
    t_test_pred = predict_logits(teacher, xte).argmax(1)
    teacher_acc = (t_test_pred == yte).float().mean().item()
    print(f"teacher test accuracy {teacher_acc:.4f}")

    runs, hists = [], []
    for seed in SEEDS:
        for name, loss_fn in VARIANTS.items():
            set_seeds(seed)
            student = SmallCNN()                    # identical initial weights for every variant
            with Energy() as e:
                hist, t = train(student, (xtr, ytr, t_train), val_d, EPOCHS, loss_fn=loss_fn,
                                seed=seed, tag=f"{name} seed {seed}")
            torch.save(student.state_dict(), MODELS_DIR / f"student_{name}_{seed}.pt")
            pred = predict_logits(student, xte).argmax(1)
            runs.append({"variant": name, "seed": seed, "test_acc": (pred == yte).float().mean().item(),
                         "val_acc": hist[-1]["val_acc"],
                         "agreement_with_teacher": (pred == t_test_pred).float().mean().item(),
                         "train_time_s": t, "train_energy_kwh": e.kwh})
            hists += [{"variant": name, "seed": seed, **h} for h in hist]
            print(f"{name:<10} seed {seed}: test acc {runs[-1]['test_acc']:.4f}", flush=True)
    runs = pd.DataFrame(runs)
    runs.to_csv(RESULTS / "distillation_runs.csv", index=False, float_format="%.5f")
    hist_df = pd.DataFrame(hists)
    hist_df.to_csv(RESULTS / "distillation_history.csv", index=False, float_format="%.5f")

    # ------------------------------------------------------------ comparison table
    xb = xte
    teacher_cost = {"params": count_params(teacher), "size_bytes": serialized_bytes(teacher.state_dict()),
                    "flops_per_image": flops_per_image(teacher), **model_latency(teacher, xb),
                    "inference_energy_kwh_per_1k": inference_energy(teacher, xb)[1]}
    student = SmallCNN()
    student.load_state_dict(torch.load(MODELS_DIR / "student_kd_manual_42.pt", weights_only=True))
    student_cost = {"params": count_params(student), "size_bytes": serialized_bytes(student.state_dict()),
                    "flops_per_image": flops_per_image(student), **model_latency(student, xb),
                    "inference_energy_kwh_per_1k": inference_energy(student, xb)[1]}
    rows = [{"model": "teacher (baseline BigCNN)", "test_acc_mean": teacher_acc, "test_acc_std": 0.0,
             "agreement_with_teacher": 1.0, "train_time_s": float("nan"), **teacher_cost}]
    for name in VARIANTS:
        r = runs[runs.variant == name]
        rows.append({"model": f"student {name}", "test_acc_mean": r.test_acc.mean(),
                     "test_acc_std": r.test_acc.std(ddof=1),
                     "agreement_with_teacher": r.agreement_with_teacher.mean(),
                     "train_time_s": r.train_time_s.mean(),
                     "train_energy_kwh": r.train_energy_kwh.mean(), **student_cost})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "distillation.csv", index=False, float_format="%.6g")
    print("\n" + table.round(4).to_string(index=False))

    # ------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for name in VARIANTS:
        m = hist_df[hist_df.variant == name].groupby("epoch")["val_acc"]
        mean, std = m.mean(), m.std(ddof=1)
        ax.plot(mean.index, mean, marker=".", label=name)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
    ax.axhline(teacher_acc, color="k", ls=":", lw=0.8, label="teacher (test)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Student training (mean ± std over 3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    sub = table.iloc[1:]
    ax.bar(sub.model.str.replace("student ", ""), sub.test_acc_mean, yerr=sub.test_acc_std, capsize=6,
           color=["tab:gray", "tab:blue", "tab:green"])
    ax.axhline(teacher_acc, color="k", ls=":", lw=0.8, label=f"teacher {teacher_acc:.4f}")
    lo = min(sub.test_acc_mean - sub.test_acc_std.fillna(0)) - 0.01
    ax.set_ylim(lo, teacher_acc + 0.01)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Student ({count_params(student):,} params) vs teacher ({teacher_cost['params']:,})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "distillation.png", dpi=130)


if __name__ == "__main__":
    main()
