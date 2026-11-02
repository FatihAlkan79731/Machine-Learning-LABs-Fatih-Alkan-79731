"""Lab 4 data pipeline. One command runs everything:

    python lab04/src/pipeline.py

ingest -> validate -> split -> clean/transform (fit on train only) -> derived-feature
experiment -> label-quality audit -> SHA256 hash + results/run_config.yaml + data_quality.png
"""
import hashlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (CATEGORICAL, CONFIG, DERIVED, FEATURES, NUMERIC, POSITIVE, PROCESSED_DIR,
                    RESULTS, TARGET)
from ingest import ingest, policy_view, sha256_of_frame
from label_quality import audit
from preprocess import build_preprocessor
from validate import to_markdown, validate

SEED = CONFIG["seed"]


def set_seeds():
    import random

    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


def plot_quality(df, train_raw, stats, feat):
    """Data-quality figure: missing values, label balance, clipping, feature experiment."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    miss = pd.Series({c: (df[c] == "?").mean() * 100 for c in CATEGORICAL})
    miss = miss[miss > 0].sort_values()
    ax[0, 0].barh(miss.index, miss.values, color="#4c72b0")
    ax[0, 0].set_xlabel("rows with '?' (%)")
    ax[0, 0].set_title("Missing values per column (full data)")

    rates = {"all": (df[TARGET] == POSITIVE).mean()}
    for col in ["sex", "race"]:
        for v, r in df.groupby(col)[TARGET].apply(lambda s: (s == POSITIVE).mean()).items():
            rates[f"{col}: {v}"] = r
    rates = pd.Series(rates) * 100
    ax[0, 1].barh(rates.index, rates.values, color="#dd8452")
    ax[0, 1].set_xlabel("share with income >50K (%)")
    ax[0, 1].set_title("Label balance by group")

    col = "hours-per-week"
    row = stats.set_index("feature").loc[col]
    ax[1, 0].hist(train_raw[col], bins=50, color="#55a868", alpha=0.8)
    for b in (row.clip_low_p1, row.clip_high_p99):
        ax[1, 0].axvline(b, color="black", ls="--", lw=1)
    ax[1, 0].set_xlabel(col)
    ax[1, 0].set_ylabel("train rows")
    ax[1, 0].set_title("Train distribution and 1st/99th percentile clip bounds")

    x = np.arange(2)
    for i, metric in enumerate(["accuracy", "f1"]):
        ax[1, 1].bar(x + i * 0.35, feat[metric], width=0.35, label=metric)
        for xi, v in zip(x + i * 0.35, feat[metric]):
            ax[1, 1].text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    ax[1, 1].set_xticks(x + 0.175, ["without derived", "with derived"])
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].legend()
    ax[1, 1].set_title("LogisticRegression baseline (test set)")

    fig.tight_layout()
    fig.savefig(RESULTS / "data_quality.png", dpi=150)


def log(lines, text):
    print(text)
    lines.append(text)


def main():
    set_seeds()
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    # 2. ingestion -----------------------------------------------------------
    raw, ing = ingest()
    log(out, f"[ingest] rows={ing['rows']} columns={ing['columns']} "
             f"(expected 48842 x 15: {'OK' if (ing['rows'], ing['columns']) == (48842, 15) else 'MISMATCH'})")
    log(out, f"[ingest] raw sha256={ing['raw_sha256']}")
    (RESULTS / "ingestion_log.json").write_text(json.dumps(ing, indent=2), encoding="utf-8")
    df = policy_view(raw)

    # 3. validation ----------------------------------------------------------
    report = validate(df, ing["dtypes"])
    report.to_csv(RESULTS / "validation_report.csv", index=False)
    (RESULTS / "validation_report.md").write_text(
        "# Validation report\n\n" + to_markdown(report) + "\n", encoding="utf-8")
    n_fail = int((report.status == "FAIL").sum())
    log(out, f"[validate] {len(report)} checks, {n_fail} failed (rows are kept, nothing dropped)")

    # 4. split before any cleaning -------------------------------------------
    train_df, test_df = train_test_split(df, test_size=CONFIG["split"]["test_size"],
                                         stratify=df[TARGET], random_state=SEED)
    n_train_raw = len(train_df)
    train_raw = train_df.copy()
    n_dup = int(train_df.duplicated().sum())
    train_df = train_df.drop_duplicates().copy()
    log(out, f"[split] train={n_train_raw} test={len(test_df)}; dropped {n_dup} exact duplicates "
             f"from train -> {len(train_df)}; test untouched")

    X_train, y_train = train_df[FEATURES], (train_df[TARGET] == POSITIVE).astype(int)
    X_test, y_test = test_df[FEATURES], (test_df[TARGET] == POSITIVE).astype(int)

    # fitted preprocessor (with derived features) ------------------------------
    prep = build_preprocessor(with_features=True).fit(X_train)
    joblib.dump(prep, RESULTS / "preprocessor.joblib")
    num = prep.named_steps["columns"].named_transformers_["num"]
    cat = prep.named_steps["columns"].named_transformers_["cat"]
    stats = pd.DataFrame({
        "feature": NUMERIC + DERIVED,
        "train_median": num.named_steps["impute"].statistics_,
        "clip_low_p1": num.named_steps["clip"].lo_,
        "clip_high_p99": num.named_steps["clip"].hi_,
        "scale_mean": num.named_steps["scale"].mean_,
        "scale_std": num.named_steps["scale"].scale_,
    })
    stats.to_csv(RESULTS / "fitted_numeric_statistics.csv", index=False, float_format="%.4f")
    pd.DataFrame({"feature": CATEGORICAL, "train_mode": cat.named_steps["impute"].statistics_}) \
        .to_csv(RESULTS / "fitted_categorical_modes.csv", index=False)

    proc_train = prep.transform(X_train).assign(**{TARGET: y_train.values})
    proc_test = prep.transform(X_test).assign(**{TARGET: y_test.values})
    proc_train.to_csv(PROCESSED_DIR / "adult_train_processed.csv", index=False,
                      lineterminator="\n", float_format="%.6f")
    proc_test.to_csv(PROCESSED_DIR / "adult_test_processed.csv", index=False,
                     lineterminator="\n", float_format="%.6f")
    h_train, h_test = sha256_of_frame(proc_train), sha256_of_frame(proc_test)
    h_all = hashlib.sha256((h_train + h_test).encode()).hexdigest()
    log(out, f"[transform] processed train {proc_train.shape}, test {proc_test.shape}")
    log(out, f"[hash] processed sha256={h_all}")

    # 5. derived-feature experiment -------------------------------------------
    rows = []
    for with_feat in (False, True):
        model = Pipeline([
            ("prep", build_preprocessor(with_features=with_feat)),
            ("clf", LogisticRegression(max_iter=CONFIG["baseline_model"]["max_iter"],
                                       random_state=SEED)),
        ]).fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"derived_features": "yes" if with_feat else "no",
                     "n_features": model.named_steps["prep"].transform(X_test.head(1)).shape[1],
                     "accuracy": accuracy_score(y_test, pred),
                     "f1": f1_score(y_test, pred)})
    feat = pd.DataFrame(rows)
    feat.to_csv(RESULTS / "feature_experiment.csv", index=False, float_format="%.4f")
    log(out, "[features]\n" + feat.round(4).to_string(index=False))

    # 6. label-quality audit --------------------------------------------------
    summary, sample, conflict, counts = audit(df)
    sample.to_csv(RESULTS / "label_audit_sample_50.csv", index=False)
    conflict.to_csv(RESULTS / "label_conflicts.csv", index=False)
    counts.rename_axis("rule").reset_index(name="rows").to_csv(
        RESULTS / "dictionary_issues_full.csv", index=False)
    (RESULTS / "label_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(out, "[labels] " + json.dumps(summary))
    log(out, "[labels] dictionary issues on full data:\n" + counts.to_string())

    # dataset statistics for the data card --------------------------------------
    parts = []
    for col in ["sex", "race", "native-country", TARGET, "workclass", "occupation"]:
        vc = df[col].value_counts(normalize=True).head(6)
        parts.append(pd.DataFrame({"column": col, "value": vc.index, "share": vc.values}))
    pd.concat(parts).to_csv(RESULTS / "dataset_stats.csv", index=False, float_format="%.4f")
    pos_by = {c: (df.groupby(c)[TARGET].apply(lambda s: (s == POSITIVE).mean()))
              for c in ["sex", "race"]}
    pd.concat(pos_by, names=["column", "value"]).rename("positive_rate").reset_index() \
        .to_csv(RESULTS / "positive_rate_by_group.csv", index=False, float_format="%.4f")

    # 7. config + hashes -------------------------------------------------------
    run = {
        "config": CONFIG,
        "data": {
            "raw_rows": ing["rows"], "raw_columns": ing["columns"],
            "raw_sha256": ing["raw_sha256"],
            "train_rows_before_dedup": n_train_raw, "train_duplicates_dropped": n_dup,
            "train_rows": len(proc_train), "test_rows": len(proc_test),
            "processed_columns": int(proc_train.shape[1]),
            "processed_train_sha256": h_train, "processed_test_sha256": h_test,
            "processed_sha256": h_all,
        },
    }
    with open(RESULTS / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run, f, sort_keys=False, allow_unicode=True)
    plot_quality(df, train_raw, stats, feat)
    (RESULTS / "pipeline_log.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nDone. Outputs in lab04/results/")


if __name__ == "__main__":
    main()
