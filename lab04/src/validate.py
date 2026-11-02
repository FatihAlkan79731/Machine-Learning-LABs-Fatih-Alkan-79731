"""Step 3: schema, range, missing-value, label and duplicate checks.

Every check is reported; no row is dropped here.
"""
import numpy as np
import pandas as pd

from config import FEATURES, MISSING_TOKEN, POSITIVE, RANGES, SCHEMA_TYPES, TARGET


def _is_int_like(s):
    v = pd.to_numeric(s, errors="coerce")
    v = v[v.notna()]
    return bool(len(v) == 0 or np.all(np.equal(np.mod(v, 1), 0)))


def validate(df, raw_dtypes):
    rows = []

    def add(check, column, ok, violations, detail):
        rows.append({"check": check, "column": column, "status": "PASS" if ok else "FAIL",
                     "violations": int(violations), "detail": detail})

    # schema: columns present
    expected = list(SCHEMA_TYPES)
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    add("schema: columns", "*", not missing_cols and not extra_cols,
        len(missing_cols) + len(extra_cols),
        f"expected 15, found {df.shape[1]}; missing={missing_cols}, extra={extra_cols}")

    # schema: types
    for col, kind in SCHEMA_TYPES.items():
        if col not in df.columns:
            continue
        raw_t = raw_dtypes.get(col if col != TARGET else "class", raw_dtypes.get(col, "?"))
        if kind == "int":
            numeric = pd.api.types.is_numeric_dtype(df[col])
            n_bad = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            int_like = _is_int_like(df[col])
            add("schema: type int", col, numeric and int_like and n_bad == 0, n_bad,
                f"raw dtype {raw_t}; all values whole numbers: {int_like}")
        else:
            is_cat = raw_t in ("category", "object")
            add("schema: type category", col, is_cat, 0 if is_cat else 1,
                f"raw dtype {raw_t}; {df[col].nunique()} distinct values")

    # ranges
    for col, (lo, hi) in RANGES.items():
        v = pd.to_numeric(df[col], errors="coerce")
        bad = ((v < lo) | (v > hi)).sum()
        add("range", col, bad == 0, bad,
            f"allowed {lo}-{hi}; observed {v.min():g}-{v.max():g}")

    # missing values ('?' is allowed but reported)
    for col in FEATURES:
        if pd.api.types.is_numeric_dtype(df[col]):
            n = int(df[col].isna().sum())
        else:
            n = int((df[col] == MISSING_TOKEN).sum())
        add("missing ('?')", col, True, n, f"{n / len(df):.2%} of rows (allowed by null policy)")

    # label values
    labels = set(df[TARGET].unique())
    allowed = {"<=50K", POSITIVE}
    bad_labels = int((~df[TARGET].isin(allowed)).sum())
    add("label values", TARGET, bad_labels == 0, bad_labels,
        f"found {sorted(labels)}; positive rate {(df[TARGET] == POSITIVE).mean():.2%}")

    # duplicates
    n_dup = int(df.duplicated().sum())
    add("duplicates: exact rows", "*", True, n_dup,
        "reported only; removed later from train split")
    n_dup_feat = int(df.duplicated(subset=FEATURES).sum())
    add("duplicates: same features", "*", True, n_dup_feat,
        "rows whose 14 features repeat an earlier row (label may differ)")

    return pd.DataFrame(rows)


def to_markdown(report):
    lines = ["| Check | Column | Status | Violations | Detail |", "|---|---|---|---|---|"]
    for _, r in report.iterrows():
        lines.append(f"| {r.check} | {r.column} | {r.status} | {r.violations} | {r.detail} |")
    return "\n".join(lines)
