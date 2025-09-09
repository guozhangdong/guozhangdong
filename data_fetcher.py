"""Mock data fetcher for examples and tests."""
from __future__ import annotations

import pandas as pd


def fetch_latest_data() -> pd.DataFrame:
    """Return a DataFrame with the latest market data.

    This mock implementation produces a single row with fixed values
    suitable for tests and demonstration.
    """
    return pd.DataFrame(
        {
            "price": [1.0],
            "volume": [100],
            "macd": [0.1],
            "bbands": [0.2],
        }
    )
