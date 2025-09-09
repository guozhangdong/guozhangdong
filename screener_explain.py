import os
import pandas as pd
import matplotlib.pyplot as plt

def build_context(df, fundamentals=None, symbol=None):
    """Placeholder build_context implementation."""
    return {
        'symbol': symbol,
        'fundamentals': fundamentals,
        'rows': len(df),
    }

def plot_price(df, sym, outdir):
    try:
        ma50 = df['close'].rolling(50).mean()
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(df['time_key'], df['close'], label='close')
        ax.plot(df['time_key'], ma50, label='sma50')
        ax.plot(df['time_key'], ema20, label='ema20')
        ax.legend()
        ax.set_title(f'{sym} price')
        fig.autofmt_xdate()
        path = os.path.join(outdir, f'{sym}_price.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception:
        return ''

def plot_rsi(df, sym, outdir):
    try:
        # recompute basic RSI14 if not present
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, 1e-9))
        rsi = 100 - 100 / (1 + rs)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(df['time_key'], rsi, label='rsi14')
        ax.axhline(30)
        ax.axhline(70)
        ax.legend()
        ax.set_title(f'{sym} rsi14')
        fig.autofmt_xdate()
        path = os.path.join(outdir, f'{sym}_rsi.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return path
    except Exception:
        return ''

def explain(config: str = './config.yaml', outdir: str = './reports', top_k: int = 10):
    """Produce an explanation report for top_k symbols.

    This placeholder implementation simply demonstrates the image generation
    hooks expected by downstream tooling.
    """
    os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
    # Dummy dataframe
    df = pd.DataFrame({'time_key': pd.date_range('2021-01-01', periods=100),
                       'close': pd.Series(range(100)).astype(float)})
    sym = 'DUMMY'
    funds = {}
    ctx = build_context(df, fundamentals=funds, symbol=sym)
    p1 = plot_price(df, sym, os.path.join(outdir, 'images'))
    p2 = plot_rsi(df, sym, os.path.join(outdir, 'images'))
    lines = []
    if p1:
        lines.append(f'![]({os.path.join("images", os.path.basename(p1))})')
    if p2:
        lines.append(f'![]({os.path.join("images", os.path.basename(p2))})')
    lines.append(f"Context: {ctx}")
    return '\n'.join(lines)

if __name__ == '__main__':
    print(explain())
