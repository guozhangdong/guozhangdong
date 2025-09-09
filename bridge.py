"""Feature bridge ensures model-ready inputs.

This module sanitises raw indicator DataFrames into a single row of
`float32` features ordered according to `config.yaml`'s
`features.cols`.  Missing columns are filled with ``0``, invalid values
(`NaN`/``inf``) are replaced with ``0`` and the result is forced into a
shape of ``(1, n)``.  A Prometheus gauge ``features_nan_rate`` records
how many values were cleaned.  If after cleaning the data is still
invalid, :class:`FeatureBridgeError` is raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import logging

import numpy as np
import pandas as pd

from metrics import features_nan_rate

LOGGER = logging.getLogger(__name__)


class FeatureBridgeError(RuntimeError):
    """Raised when the feature bridge cannot produce a valid feature row."""


@dataclass
class BridgeReport:
    missing_cols: List[str]
    nan_ratio: float
    shape: tuple
    dtype: str


def _clean_dataframe(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            LOGGER.warning("missing column %s, filling with 0", col)
            df[col] = 0.0
    df = df[feature_cols]
    # Coerce to numeric and replace non-finite values
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return df


def build_latest_feature_row(df: pd.DataFrame, feature_cols: Iterable[str]) -> np.ndarray:
    """Return the latest feature row as ``float32``.

    Parameters
    ----------
    df:
        DataFrame containing raw feature columns.
    feature_cols:
        Ordered list of required feature columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(1, n)`` with ``float32`` dtype.

    Raises
    ------
    FeatureBridgeError
        If the resulting array cannot be coerced into the required
        shape or dtype.
    """
    feature_cols = list(feature_cols)
    cleaned = _clean_dataframe(df, feature_cols)
    missing_cols = [c for c in feature_cols if c not in df.columns]

    latest = cleaned.tail(1)
    arr = latest.to_numpy(dtype=np.float32, copy=False)
    nan_count = np.isnan(arr).sum() + np.isinf(arr).sum()
    if nan_count:
        LOGGER.warning("found %d NaN/inf values, filled with 0", nan_count)
    arr = np.nan_to_num(arr, copy=False)
    features_nan_rate.set(nan_count / arr.size)

    if arr.shape != (1, len(feature_cols)):
        raise FeatureBridgeError(
            f"feature row shape {arr.shape} != (1, {len(feature_cols)})"
        )
    if arr.dtype != np.float32:
        raise FeatureBridgeError(f"feature row dtype {arr.dtype} != float32")

    return arr
