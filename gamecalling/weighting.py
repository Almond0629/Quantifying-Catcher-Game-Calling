"""Integer game weights for the bootstrap.

A resampled game drawn m times can be represented either by m copies of its rows
or by one copy with weight `w` = m. Every count, mean and cross-product in the
pipeline uses `w` when present (1 otherwise), so the two representations give
identical fits. `w_hyper` (default: `w`) is the weight used only to estimate
hyperparameters (pooling concentrations and ranking variance components); the
experimental "firstcopy" bootstrap sets it to 1 for every drawn game.
"""

import numpy as np
import pandas as pd


def w(df: pd.DataFrame) -> np.ndarray:
    return df["w"].to_numpy(float) if "w" in df else np.ones(len(df))


def w_hyper(df: pd.DataFrame) -> np.ndarray:
    return df["w_hyper"].to_numpy(float) if "w_hyper" in df else w(df)


def wmean(values, weights):
    weights = np.asarray(weights, float)
    return float(np.sum(np.asarray(values, float) * weights) / weights.sum())
