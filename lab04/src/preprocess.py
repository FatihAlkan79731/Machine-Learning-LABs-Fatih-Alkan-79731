"""Steps 4-5: the fitted preprocessing pipeline and the derived features.

Everything that learns a statistic (median, mode, percentiles, mean/std,
category list) is fitted on the training split only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL, CONFIG, DERIVED, MISSING_TOKEN, NUMERIC


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Row-wise derived features. Nothing is learned, so it cannot leak."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(list(names) + DERIVED, dtype=object)

    def transform(self, X):
        X = X.copy()
        X["capital_net"] = X["capital-gain"] - X["capital-loss"]
        X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
        return X


class PercentileClipper(TransformerMixin, BaseEstimator):
    """Clip every column to the [lower, upper] percentiles seen during fit."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.names_ = list(X.columns) if hasattr(X, "columns") else None
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.names_ is not None:
            return np.asarray(self.names_, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_preprocessor(with_features=True):
    numeric = NUMERIC + (DERIVED if with_features else [])
    clip = CONFIG["clipping"]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper(clip["lower_percentile"], clip["upper_percentile"])),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(missing_values=MISSING_TOKEN, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, CATEGORICAL),
    ])
    steps = ([("derive", FeatureAdder())] if with_features else []) + [("columns", columns)]
    return Pipeline(steps).set_output(transform="pandas")
