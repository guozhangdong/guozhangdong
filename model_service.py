"""Trivial model service for demonstration purposes."""
from __future__ import annotations

import numpy as np


def predict_proba_one(x: np.ndarray) -> float:
    """Return a dummy probability for the given feature row."""
    return float(1 / (1 + np.exp(-x.sum())))
