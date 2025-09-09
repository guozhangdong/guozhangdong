"""Prometheus metrics for feature bridge."""
from prometheus_client import Gauge

# Gauge for percentage of NaN/inf replaced in feature row
features_nan_rate: Gauge = Gauge("features_nan_rate", "Rate of NaN/inf features in latest row")
