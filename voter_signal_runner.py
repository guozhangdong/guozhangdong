"""Stub voter signal runner with metrics and alert hooks.

This module demonstrates how a score based "voter" strategy could expose
Prometheus metrics and trigger alerts when risk thresholds are exceeded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from prometheus_client import Gauge
from broker import place_order  # placeholder import for real trade integration
from alerts import send_alert

# Core gauges

g_sig = Gauge('voter_signal', 'Voter signal: -1 sell, 0 hold, 1 buy', ['symbol'])
g_score = Gauge('voter_score', 'Voter weighted score', ['symbol'])
g_rules_eval = Gauge('voter_rules_evaluated', 'Rules evaluated count', ['symbol'])
g_rules_pass = Gauge('voter_rules_passed', 'Rules passed count', ['symbol'])
g_fund_missing = Gauge('fundamentals_missing_fields', 'Missing fundamentals fields count', ['symbol'])
g_pnl = Gauge('voter_unrealized_pnl', 'Unrealized PnL', ['symbol'])

def run_once(sym: str, score: float, pnl: float, cfg: Dict[str, Any]) -> None:
    """Update metrics for ``sym`` and trigger alerts if thresholds breached."""
    sig = 'BUY' if score > 0 else ('SELL' if score < 0 else 'HOLD')
    g_sig.labels(symbol=sym).set({'BUY': 1, 'SELL': -1}.get(sig, 0))
    g_score.labels(symbol=sym).set(score)
    g_rules_eval.labels(symbol=sym).set(0)
    g_rules_pass.labels(symbol=sym).set(0)
    g_fund_missing.labels(symbol=sym).set(0)
    g_pnl.labels(symbol=sym).set(pnl)

    al = cfg.get('alerts', {})
    min_pnl = float(al.get('min_unrealized_pnl', -999999))
    min_score = float(al.get('min_voter_score', -9e9))
    if al.get('enabled', False):
        if pnl <= min_pnl:
            send_alert('PnLDrop', {'symbol': sym, 'pnl': pnl}, cfg)
        if score <= min_score:
            send_alert('ScoreTooLow', {'symbol': sym, 'score': score}, cfg)

def main():
    cfg = {}
    cfg_path = Path('config.yaml')
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
    # Example demo call
    run_once('DEMO', 0.5, -100.0, cfg)

if __name__ == '__main__':
    main()
