"""Pipeline configuration. Written to results/config.yaml together with the data hashes."""
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
RESULTS = LAB_DIR / "results"

CONFIG = {
    "dataset": {"name": "adult", "version": 2, "source": "OpenML (fetch_openml)"},
    "seed": 42,
    "split": {"test_size": 0.30, "stratify": "income"},
    "imputation": {"numeric": "median", "categorical": "most_frequent"},
    "clipping": {"lower_percentile": 1, "upper_percentile": 99},
    "duplicates": "drop exact duplicates from train only",
    "derived_features": ["capital_net = capital-gain - capital-loss",
                         "hours_per_age = hours-per-week / (age - 16)"],
    "baseline_model": {"type": "LogisticRegression", "max_iter": 1000},
    "label_audit": {"sample_size": 50},
}

TARGET = "income"
POSITIVE = ">50K"

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
DERIVED = ["capital_net", "hours_per_age"]

# schema from the lab manual
SCHEMA_TYPES = {c: "int" for c in NUMERIC}
SCHEMA_TYPES.update({c: "category" for c in CATEGORICAL})
SCHEMA_TYPES[TARGET] = "category"

RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

MISSING_TOKEN = "?"

# data dictionary: education level -> education-num
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}
