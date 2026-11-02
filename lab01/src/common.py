"""Shared helpers for Lab 1: seeds, data split and model definitions."""
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"

MODEL_NAMES = {
    "logreg": "LogisticRegression",
    "rf": "RandomForestClassifier",
}


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split():
    """Breast Cancer Wisconsin, 70/30 stratified split."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def make_model(key):
    if key == "logreg":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if key == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=SEED)
    raise ValueError(f"unknown model: {key}")
