"""Shared helpers for Lab 3: seeds, paths and Fashion-MNIST loaders."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH = 128
LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
DATA = ROOT / "data"
RESULTS = LAB_DIR / "results"
DEVICE = torch.device("cpu")


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets():
    """55k train / 5k validation from the official 60k train set, official 10k test."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(root=str(DATA), train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=str(DATA), train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def get_loaders():
    train, val, test = get_datasets()
    g = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(train, batch_size=BATCH, shuffle=True, generator=g, num_workers=0),
        DataLoader(val, batch_size=BATCH, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=0),
    )
