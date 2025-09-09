import pandas as pd

import data_fetcher
import model_service
import stream_runner
from metrics import features_nan_rate


def test_stream_runner(monkeypatch):
    monkeypatch.setattr(data_fetcher, "fetch_latest_data", lambda: pd.DataFrame({
        "price": [1.0],
        "volume": [2.0],
        "macd": [3.0],
        "bbands": [4.0],
    }))
    monkeypatch.setattr(model_service, "predict_proba_one", lambda x: 0.5)
    prob = stream_runner.run_once(["price", "volume", "macd", "bbands"])
    assert prob == 0.5
    assert features_nan_rate._value.get() == 0.0
