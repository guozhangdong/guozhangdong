# futu_autotrade/backtest_lib.py
from __future__ import annotations
import math
import pandas as pd
import numpy as np


def cost_per_notional_bps(atr_pct: float, spread_bps: float = 0.0, slip_bps: float = 0.0, slip_atr_mult: float = 0.0) -> float:
    """Return total bps cost per one-side trade notional (commission + half-spread + slippage).
    atr_pct: ATR / price (as proportion), e.g., 0.01 for 1%.\n
    """
    # half-spread modeled as spread_bps/2
    return float((spread_bps/2.0) + slip_bps + (slip_atr_mult * (atr_pct*1e4)))


def equity_metrics(equity: pd.Series, rf: float = 0.0) -> dict:
    if len(equity) < 2:
        return {'CAGR':0,'Sharpe':0,'MDD':0,'Vol':0}
    rets = equity.pct_change().fillna(0.0)
    days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[0], 'to_pydatetime') else len(equity)
    years = max(1e-9, days/365.25)
    cagr = (equity.iloc[-1] / max(1e-9, equity.iloc[0])) ** (1/years) - 1
    vol = rets.std() * (252 ** 0.5)
    sharpe = 0.0 if vol == 0 else (rets.mean() * 252 - rf) / vol
    peak = equity.cummax()
    dd = equity/peak - 1.0
    mdd = dd.min()
    return {'CAGR': float(cagr), 'Sharpe': float(sharpe), 'MDD': float(mdd), 'Vol': float(vol)}


def ts_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity/peak - 1.0
