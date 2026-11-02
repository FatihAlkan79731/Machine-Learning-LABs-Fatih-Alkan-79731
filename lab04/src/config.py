"""Paths, schema and the pipeline configuration (loaded from lab04/config.yaml)."""
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
PROCESSED_DIR = DATA / "processed"
RESULTS = LAB_DIR / "results"

CONFIG_FILE = LAB_DIR / "config.yaml"
with open(CONFIG_FILE, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

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
