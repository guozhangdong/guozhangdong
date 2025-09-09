# futu_autotrade/backtest_voter.py
"""Walk-forward backtest for rule-voting strategy.
Usage:
  python backtest_voter.py --config ./config.yaml --rules ./rules.yaml --symbol US.AAPL --bars 2000 --cost_bps 5
Outputs (./reports):
  - backtest_trades.csv
  - backtest_equity.csv
  - backtest_summary.md
  - images/equity_curve.png, images/drawdown.png
"""
import argparse, os, math, yaml
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from data_fetcher import fetch_kline
from formulas import build_context, load_fundamentals, safe_eval
from strategies import vote_signal

def load_cfg(path='./config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_rules(path: str | None):
    import yaml
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            dat = yaml.safe_load(f)
        rules = dat.get('rules', [])
    else:
        rules = [
            {'rule': 'ema_20 > ema_50', 'weight': 1.0},
            {'rule': 'macd_hist > 0', 'weight': 0.5},
            {'rule': 'rsi_14 < 70', 'weight': 0.2},
            {'rule': 'close > sma_50', 'weight': 0.3},
        ]
    return {r['rule']: float(r.get('weight', 1.0)) for r in rules}

def metrics(equity: pd.Series, rf: float = 0.0) -> dict:
    if len(equity) < 2:
        return {'CAGR': 0, 'Sharpe': 0, 'MDD': 0, 'WinRate': 0, 'Trades': 0}
    rets = equity.pct_change().fillna(0.0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if hasattr(equity.index[0], 'to_pydatetime') else len(equity) / 252 / 24 / 60
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = (rets.mean() - rf/252) / (rets.std() + 1e-12) * (252 ** 0.5)
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    mdd = dd.min()
    return {'CAGR': float(cagr), 'Sharpe': float(sharpe), 'MDD': float(mdd)}

def backtest(df: pd.DataFrame, rules: dict, funds: dict, symbol: str, cost_bps: float = 5.0):
    pos = 0  # -1,0,1 (long/flat/short). Here we use long/flat only for simplicity.
    cash = 1.0
    shares = 0.0
    trades = []
    equity = []
    times = []
    for i in range(60, len(df)):  # warmup
        window = df.iloc[:i+1].copy()
        ctx = build_context(window, fundamentals=funds, symbol=symbol)
        score = 0.0
        for expr, w in rules.items():
            try:
                if safe_eval(expr, ctx):
                    score += w
            except Exception:
                pass
        sig = 'BUY' if score > 0 else ('SELL' if score < 0 else 'HOLD')
        px = float(window['close'].iloc[-1])
        ts = pd.to_datetime(window['time_key'].iloc[-1])
        if sig == 'BUY' and pos == 0:
            qty = cash / px
            fee = qty * px * cost_bps / 10000.0
            qty = (cash - fee) / px
            shares += qty
            cash -= qty * px + fee
            pos = 1
            trades.append({'time': ts, 'side': 'BUY', 'price': px, 'qty': qty, 'score': score})
        elif sig == 'SELL' and pos == 1:
            fee = shares * px * cost_bps / 10000.0
            cash += shares * px - fee
            trades.append({'time': ts, 'side': 'SELL', 'price': px, 'qty': shares, 'score': score})
            shares = 0.0
            pos = 0
        eq = cash + shares * px
        equity.append(eq)
        times.append(ts)
    eq_series = pd.Series(equity, index=pd.to_datetime(times))
    return pd.DataFrame(trades), eq_series

def plot_series(s: pd.Series, title: str, path: str):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(s.index, s.values, label=title)
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='./config.yaml')
    ap.add_argument('--rules', default='./rules.yaml')
    ap.add_argument('--symbol', default=None)
    ap.add_argument('--bars', type=int, default=2000)
    ap.add_argument('--cost_bps', type=float, default=5.0)
    ap.add_argument('--out', default='./reports')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    futu = cfg['futu']
    sym = args.symbol or (cfg.get('symbols')[0] if cfg.get('symbols') else futu['symbol'])
    rules = load_rules(args.rules)
    fund_path = os.path.join(os.path.dirname(args.config), 'data', 'fundamentals.csv')
    if not os.path.exists(fund_path):
        fund_path = './data/fundamentals.csv'
    funds = load_fundamentals(fund_path)

    df, err = fetch_kline(futu['host'], futu['port'], sym, futu['ktype'], args.bars, futu['autype'])
    if err:
        print('[WARN]', err)
    trades, equity = backtest(df, rules, funds, sym, args.cost_bps)

    os.makedirs(args.out, exist_ok=True)
    trades_path = os.path.join(args.out, 'backtest_trades.csv')
    eq_path = os.path.join(args.out, 'backtest_equity.csv')
    trades.to_csv(trades_path, index=False)
    equity.to_csv(eq_path, header=['equity'])

    m = metrics(equity)
    md = [
        '# Backtest Summary',
        '',
        f'- Symbol: **{sym}**',
        f'- Bars: {len(df)}',
        f'- Trades: {len(trades)}',
        f'- CAGR: {m.get("CAGR", 0):.2%}',
        f'- Sharpe: {m.get("Sharpe", 0):.2f}',
        f'- Max Drawdown: {m.get("MDD", 0):.2%}',
    ]
    with open(os.path.join(args.out, 'backtest_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    plot_series(equity, f'Equity Curve ({sym})', os.path.join(args.out, 'images', 'equity_curve.png'))
    peak = equity.cummax()
    dd = equity / peak - 1.0
    plot_series(dd, f'Drawdown ({sym})', os.path.join(args.out, 'images', 'drawdown.png'))

    print('Saved backtest outputs to', args.out)

if __name__ == '__main__':
    main()
