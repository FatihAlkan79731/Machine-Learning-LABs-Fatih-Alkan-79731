"""Step 2: ingestion. Reads the raw data and does not change it."""
import hashlib

import pandas as pd
from sklearn.datasets import fetch_openml

from config import CATEGORICAL, DATA, MISSING_TOKEN, RAW_DIR, TARGET


def sha256_of_frame(df):
    """Hash of a canonical CSV serialization (fixed column order, '\\n' line endings)."""
    csv = df.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def ingest():
    """Download (or load from cache) Adult v2 from OpenML and keep an untouched copy."""
    bunch = fetch_openml(name="adult", version=2, as_frame=True, data_home=str(DATA / "openml"))
    raw = bunch.frame.copy()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_DIR / "adult_raw.csv", index=False, lineterminator="\n")
    log = {
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "column_names": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "raw_sha256": sha256_of_frame(raw),
    }
    return raw, log


def policy_view(raw):
    """Same rows and values, expressed with the lab's null policy.

    OpenML already stores '?' as a missing value (NaN). Here missing categorical
    values are written back as '?', categories become plain strings and the target
    column 'class' is called 'income'. No row is removed and no value is changed.
    """
    df = raw.copy()
    if "class" in df.columns and TARGET not in df.columns:
        df = df.rename(columns={"class": TARGET})
    for c in CATEGORICAL + [TARGET]:
        df[c] = df[c].astype(object).where(df[c].notna(), MISSING_TOKEN).astype(str).str.strip()
    return df
