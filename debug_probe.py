"""Run a one-shot feature extraction and output diagnostics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

import data_fetcher
from bridge import FeatureBridgeError, build_latest_feature_row

LOGGER = logging.getLogger(__name__)


def run(config_path: str) -> None:
    """Execute the debug probe.

    Parameters
    ----------
    config_path: str
        Path to the YAML configuration file.
    """
    with open(config_path, "r", encoding="utf8") as fh:
        config = yaml.safe_load(fh)
    feature_cols = config["features"]["cols"]

    raw_df = data_fetcher.fetch_latest_data()
    try:
        X = build_latest_feature_row(raw_df, feature_cols)
    except FeatureBridgeError as err:
        LOGGER.exception("feature bridge failed: %s", err)
        raise

    report: Dict[str, Any] = {
        "columns": feature_cols,
        "dtype": str(X.dtype),
        "shape": X.shape,
    }
    Path("debug_X.npy").write_bytes(X.tobytes())
    with open("debug_report.json", "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)
    LOGGER.info("debug probe complete: %s", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.config)
