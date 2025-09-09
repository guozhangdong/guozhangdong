# futu_autotrade/backtest_batch.py
"""Batch backtest for rule-voting strategy across multiple symbols.
- Per-symbol walk-forward backtest (long/flat)
- Equal-weight portfolio curve
- HTML report with embedded charts and trade markers\n
Usage:
  python backtest_batch.py --config ./config.yaml --rules ./rules.yaml --bars 3000 --out ./reports/batch
"""
import argparse, os, io, base64, yaml
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from data_fetcher import fetch_kline
from formulas import build_context, load_fundamentals, safe_eval
from strategies import vote_signal
from backtest_lib import cost_per_notional_bps, equity_metrics, ts_drawdown

DEF_RULES = [
    {'rule': 'ema_20 > ema_50', 'weight': 1.0},
    {'rule': 'macd_hist > 0 and rsi_14 < 70', 'weight': 0.6},
    {'rule': 'close > sma_50', 'weight': 0.3},
    {'rule': 'pe < 30', 'weight': 0.2},
    {'rule': 'roe > 0.1', 'weight': 0.2},
]

def load_cfg(path='./config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_rules(path: str | None):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            dat = yaml.safe_load(f)
        rules = dat.get('rules', DEF_RULES)
    else:
        rules = DEF_RULES
    return {r['rule']: float(r.get('weight', 1.0)) for r in rules}

def to_base64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def backtest_symbol(df: pd.DataFrame, rules: dict, funds: dict, symbol: str,
                    spread_bps: float, slip_bps: float, slip_atr_mult: float):
    # prepare ATR%
    high = df['high']; low = df['low']; close = df['close']
    tr = (high-low).abs().combine((high-close.shift()).abs(), max).combine((low-close.shift()).abs(), max)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close).fillna(0.0)

    cash = 1.0; shares = 0.0; pos = 0
    trades = []; equity = []; times = []
    for i in range(60, len(df)):
        window = df.iloc[:i+1]
        ctx = build_context(window, fundamentals=funds, symbol=symbol)
        score = 0.0
        for expr, w in rules.items():
            try:
                if safe_eval(expr, ctx):
                    score += w
            except Exception:
                pass
        sig = 'BUY' if score>0 else ('SELL' if score<0 else 'HOLD')
        px = float(window['close'].iloc[-1])
        ts = pd.to_datetime(window['time_key'].iloc[-1])
        # dynamic bps by ATR
        bps = cost_per_notional_bps(float(atr_pct.iloc[i]), spread_bps=spread_bps, slip_bps=slip_bps, slip_atr_mult=slip_atr_mult)
        if sig == 'BUY' and pos == 0:
            fee = cash * (bps/1e4)
            qty = (cash - fee) / px
            shares += qty; cash -= qty*px + fee
            pos = 1; trades.append({'time': ts, 'side':'BUY', 'price': px, 'qty': qty, 'bps': bps, 'score': score})
        elif sig == 'SELL' and pos == 1:
            fee = shares * px * (bps/1e4)
            cash += shares*px - fee
            trades.append({'time': ts, 'side':'SELL', 'price': px, 'qty': shares, 'bps': bps, 'score': score})
            shares = 0.0; pos = 0
        equity.append(cash + shares*px); times.append(ts)
    eq = pd.Series(equity, index=pd.to_datetime(times))
    return pd.DataFrame(trades), eq

def plot_price_with_trades(df: pd.DataFrame, trades: pd.DataFrame, title: str):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(df['time_key'].iloc[-len(trades)*2-120:], df['close'].iloc[-len(trades)*2-120:], label='close')
    # mark trades
    for _, t in trades.iterrows():
        color = 'g' if t['side']=='BUY' else 'r'
        ax.scatter([t['time']], [t['price']], marker='^' if t['side']=='BUY' else 'v')
    ax.set_title(title); ax.legend(); fig.autofmt_xdate()
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='./config.yaml')
    ap.add_argument('--rules', default='./rules.yaml')
    ap.add_argument('--bars', type=int, default=3000)
    ap.add_argument('--out', default='./reports/batch')
    ap.add_argument('--spread_bps', type=float, default=1.0)
    ap.add_argument('--slip_bps', type=float, default=1.0)
    ap.add_argument('--slip_atr_mult', type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_cfg(args.config); futu = cfg['futu']
    symbols = cfg.get('symbols') or [futu['symbol']]
    rules = load_rules(args.rules)
    funds = load_fundamentals(os.path.join(os.path.dirname(args.config), 'data', 'fundamentals.csv') if os.path.exists(os.path.join(os.path.dirname(args.config), 'data', 'fundamentals.csv')) else './data/fundamentals.csv')

    sym_summaries = []; portfolio = None; charts = []
    for sym in symbols:
        df, err = fetch_kline(futu['host'], futu['port'], sym, futu['ktype'], args.bars, futu['autype'])
        if err: print('[WARN]', sym, err)
        trades, equity = backtest_symbol(df, rules, funds, sym, args.spread_bps, args.slip_bps, args.slip_atr_mult)
        m = equity_metrics(equity)
        # save csv
        trades.to_csv(os.path.join(args.out, f'{sym}_trades.csv'), index=False)
        equity.to_csv(os.path.join(args.out, f'{sym}_equity.csv'), header=['equity'])
        # chart
        fig = plot_price_with_trades(df, trades.tail(100), f'{sym} price & trades')
        b64 = to_base64_png(fig)
        charts.append((sym, b64))
        sym_summaries.append({'symbol': sym, 'CAGR': m['CAGR'], 'Sharpe': m['Sharpe'], 'MDD': m['MDD']})

        # portfolio (equal weight on overlapping index)
        portfolio = equity if portfolio is None else portfolio.add(equity, fill_value=np.nan)

    if portfolio is not None and len(symbols)>0:
        portfolio = (portfolio / len(symbols)).dropna()
        mP = equity_metrics(portfolio)
    else:
        mP = {'CAGR':0,'Sharpe':0,'MDD':0}

    # HTML report
    rows = ''.join([f"<tr><td>{s['symbol']}</td><td>{s['CAGR']:.2%}</td><td>{s['Sharpe']:.2f}</td><td>{s['MDD']:.2%}</td></tr>" for s in sym_summaries])
    imgs = ''.join([f"<h3>{sym}</h3><img src='data:image/png;base64,{b64}' style='max-width:900px'/>" for sym,b64 in charts])

    # portfolio charts
    import matplotlib.pyplot as plt
    def to_b64_series(series, title):
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(series.index, series.values); ax.set_title(title); fig.autofmt_xdate()
        import io, base64
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, bbox_inches='tight'); plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('ascii')
    if portfolio is not None:
        b64_eq = to_b64_series(portfolio, 'Portfolio Equity')
        dd = ts_drawdown(portfolio)
        b64_dd = to_b64_series(dd, 'Portfolio Drawdown')
        portfolio_html = f"<h2>Portfolio (Equal-weight)</h2><p>CAGR {mP['CAGR']:.2%} | Sharpe {mP['Sharpe']:.2f} | MDD {mP['MDD']:.2%}</p><img src='data:image/png;base64,{b64_eq}'/><img src='data:image/png;base64,{b64_dd}'/>"
    else:
        portfolio_html = ''

    html = f"""
    <html><head><meta charset='utf-8'><title>Batch Backtest Report</title></head>
    <body>
    <h1>Batch Backtest Report</h1>
    <p>Symbols: {', '.join(symbols)}</p>
    <h2>Per-Symbol Metrics</h2>
    <table border='1' cellspacing='0' cellpadding='6'>
      <tr><th>Symbol</th><th>CAGR</th><th>Sharpe</th><th>Max Drawdown</th></tr>
      {rows}
    </table>
    {portfolio_html}
    <h2>Recent Trades (Charts)</h2>
    {imgs}
    </body></html>
    """
    with open(os.path.join(args.out, 'backtest_summary.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print('Saved report to', os.path.join(args.out, 'backtest_summary.html'))

if __name__ == '__main__':
    main()
