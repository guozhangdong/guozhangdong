import numpy as np
import pandas as pd

from bridge import build_latest_feature_row


FEATURE_COLS = ["price", "volume", "macd", "bbands"]


def test_bridge_cleaning_handles_missing_and_nan():
    df = pd.DataFrame({
        "price": [1.0],
        "volume": [np.nan],
        "macd": ["3"],
        # bbands missing
    })
    arr = build_latest_feature_row(df, FEATURE_COLS)
    assert arr.shape == (1, len(FEATURE_COLS))
    assert arr.dtype == np.float32
    assert arr[0, 1] == 0.0  # NaN replaced
    assert arr[0, 3] == 0.0  # missing column filled
