"""Simple stream runner tying together data fetch, bridge and model."""
from __future__ import annotations

import logging
from typing import Iterable

import yaml
from prometheus_client import start_http_server

import data_fetcher
from bridge import FeatureBridgeError, build_latest_feature_row
from metrics import features_nan_rate
import model_service

LOGGER = logging.getLogger(__name__)


def run_once(feature_cols: Iterable[str]) -> float:
    """Fetch data, build feature row and run the model."""
    raw_df = data_fetcher.fetch_latest_data()
    LOGGER.debug("raw columns: %s", list(raw_df.columns))
    try:
        X = build_latest_feature_row(raw_df, feature_cols)
    except FeatureBridgeError as err:
        LOGGER.warning("feature bridge failed: %s", err)
        raise
    LOGGER.debug("feature row shape=%s dtype=%s", X.shape, X.dtype)
    return model_service.predict_proba_one(X)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with open("config.yaml", "r", encoding="utf8") as fh:
        config = yaml.safe_load(fh)
    feature_cols = config["features"]["cols"]

    start_http_server(8000)
    prob = run_once(feature_cols)
    LOGGER.info("prediction %.3f, nan rate %.3f", prob, features_nan_rate._value.get())


if __name__ == "__main__":
    main()
