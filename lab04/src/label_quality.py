"""Step 6: label-quality audit.

1. Duplicate-conflict audit: rows with identical 14 features but different income.
2. 50 random rows re-checked against the data dictionary for impossible or
   inconsistent values. The sample is saved so it can be reviewed by hand.
"""
import pandas as pd

from config import CONFIG, EDUCATION_NUM, FEATURES, MISSING_TOKEN, RANGES, TARGET


def duplicate_conflicts(df):
    g = df.groupby(FEATURES, dropna=False)[TARGET].agg(["nunique", "size"]).reset_index()
    dup_groups = g[g["size"] > 1]
    conflict = dup_groups[dup_groups["nunique"] > 1]
    return {
        "duplicate_feature_groups": int(len(dup_groups)),
        "rows_in_duplicate_groups": int(dup_groups["size"].sum()),
        "conflicting_groups": int(len(conflict)),
        "rows_in_conflicting_groups": int(conflict["size"].sum()),
    }, conflict


def dictionary_issues(row):
    """Return (rule, message) for every data-dictionary rule this row breaks."""
    issues = []
    for col, (lo, hi) in RANGES.items():
        if not lo <= row[col] <= hi:
            issues.append(("range", f"{col} outside {lo}-{hi}"))
    edu = row["education"]
    if edu in EDUCATION_NUM and EDUCATION_NUM[edu] != row["education-num"]:
        issues.append(("education mismatch", f"education '{edu}' but education-num {row['education-num']}"))
    if row["capital-gain"] > 0 and row["capital-loss"] > 0:
        issues.append(("gain and loss both > 0", "capital-gain and capital-loss both > 0"))
    if row["relationship"] == "Husband" and row["sex"] == "Female":
        issues.append(("relationship vs sex", "relationship Husband but sex Female"))
    if row["relationship"] == "Wife" and row["sex"] == "Male":
        issues.append(("relationship vs sex", "relationship Wife but sex Male"))
    if row["relationship"] in ("Husband", "Wife") and not str(row["marital-status"]).startswith("Married"):
        issues.append(("relationship vs marital-status",
                       f"relationship {row['relationship']} but marital-status {row['marital-status']}"))
    if row["workclass"] == "Never-worked" and row["occupation"] != MISSING_TOKEN:
        issues.append(("never-worked with occupation", "workclass Never-worked but occupation given"))
    if row[TARGET] not in ("<=50K", ">50K"):
        issues.append(("label", f"invalid label {row[TARGET]}"))
    return issues


def audit(df, seed=CONFIG["seed"]):
    summary, conflict = duplicate_conflicts(df)

    n = CONFIG["label_audit"]["sample_size"]
    sample = df.sample(n=n, random_state=seed).copy()
    sample["issues"] = sample.apply(lambda r: "; ".join(m for _, m in dictionary_issues(r)), axis=1)
    summary["sample_rows_checked"] = n
    summary["sample_rows_with_issues"] = int((sample["issues"] != "").sum())

    full_issues = df.apply(dictionary_issues, axis=1)
    summary["full_rows_with_issues"] = int(full_issues.map(len).gt(0).sum())
    counts = pd.Series([rule for lst in full_issues for rule, _ in lst], dtype=object).value_counts()
    return summary, sample, conflict, counts
