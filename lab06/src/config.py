"""Load lab06/config.yaml into one attribute-style config object and apply overrides.

    cfg = load_config()                                   # defaults
    cfg = load_config(overrides=["optimizer.name=sgd", "optimizer.lr=0.05"])
    cfg.optimizer.lr  -> 0.05
"""
import copy
from pathlib import Path

import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
RESULTS = LAB_DIR / "results"
DEFAULT_CONFIG = LAB_DIR / "config.yaml"


class Config(dict):
    """A dict whose keys can also be read as attributes (nested dicts become Configs)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        """Plain nested dicts (YAML's safe_dump and pickling need no custom class)."""
        return {k: v.to_dict() if isinstance(v, Config) else copy.deepcopy(v) for k, v in self.items()}


def _to_config(d):
    return Config({k: _to_config(v) if isinstance(v, dict) else v for k, v in d.items()})


def apply_override(cfg, item):
    """item = 'section.key=value'; the value is parsed with YAML (so 0.05 -> float, true -> bool)."""
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got {item!r}")
    *path, last = key.strip().split(".")
    node = cfg
    for p in path:
        if p not in node:
            raise KeyError(f"unknown config section {p!r} in {item!r}")
        node = node[p]
    if last not in node:
        raise KeyError(f"unknown config key {key!r}")
    node[last] = yaml.safe_load(raw)


def load_config(path=DEFAULT_CONFIG, overrides=()):
    with open(path, encoding="utf-8") as f:
        cfg = _to_config(yaml.safe_load(f))
    for item in overrides or ():
        apply_override(cfg, item)
    return cfg


def flatten(d, prefix=""):
    """{'optimizer': {'lr': 0.001}} -> {'optimizer.lr': 0.001} (for MLflow params)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
