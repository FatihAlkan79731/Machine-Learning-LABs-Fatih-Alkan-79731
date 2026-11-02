"""Shared helpers for Lab 2: seeds and the digits data."""
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS = LAB_DIR / "results"


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def one_hot(y, n_classes=10):
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_data():
    """Digits scaled to [0, 1], one-hot labels, 70/30 stratified split."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test)
