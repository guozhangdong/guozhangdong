import json
from pathlib import Path

import pandas as pd

import debug_probe
import data_fetcher


def test_debug_probe_outputs(tmp_path, monkeypatch):
    def fake_fetch() -> pd.DataFrame:
        return pd.DataFrame({"price": [1.0], "volume": [2.0], "macd": [3.0], "bbands": [4.0]})
    monkeypatch.setattr(data_fetcher, "fetch_latest_data", fake_fetch)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("features:\n  cols: [price, volume, macd, bbands]\n")
    monkeypatch.chdir(tmp_path)

    debug_probe.run(str(cfg))

    assert Path("debug_X.npy").exists()
    report = json.loads(Path("debug_report.json").read_text())
    assert report["dtype"] == "float32"
    assert report["shape"] == [1, 4] or tuple(report["shape"]) == (1, 4)
