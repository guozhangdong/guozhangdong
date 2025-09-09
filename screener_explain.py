# futu_autotrade/screener_explain.py
"""Generate explanation report for rules that passed, with simple charts.
Usage:
  python screener_explain.py --config ./config.yaml --out ./reports --top 10
Outputs:
  - reports/screener_explain.md
  - reports/images/<symbol>_price.png, <symbol>_rsi.png
"""
import os
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from data_fetcher import fetch_kline
from formulas import build_context, load_fundamentals, names_required


def load_cfg(path: str):
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def val_of(name: str, ctx: dict):
    """Return a float value from context if possible."""
    v = ctx.get(name, None)
    if hasattr(v, "iloc"):
        try:
            return float(v.iloc[-1])
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return float(v)
    except Exception:  # pragma: no cover - defensive
        return None


def plot_price(df: pd.DataFrame, sym: str, outdir: str) -> str:
    """Plot price with simple indicators."""
    try:
        os.makedirs(outdir, exist_ok=True)
        ma50 = df["close"].rolling(50).mean()
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(df["time_key"], df["close"], label="close")
        ax.plot(df["time_key"], ma50, label="sma50")
        ax.plot(df["time_key"], ema20, label="ema20")
        ax.legend()
        ax.set_title(f"{sym} price")
        fig.autofmt_xdate()
        path = os.path.join(outdir, f"{sym}_price.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - defensive
        return ""


def plot_rsi(df: pd.DataFrame, sym: str, outdir: str) -> str:
    """Plot a basic RSI chart."""
    try:
        os.makedirs(outdir, exist_ok=True)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, 1e-9))
        rsi = 100 - 100 / (1 + rs)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(df["time_key"], rsi, label="rsi14")
        ax.axhline(30)
        ax.axhline(70)
        ax.legend()
        ax.set_title(f"{sym} rsi14")
        fig.autofmt_xdate()
        path = os.path.join(outdir, f"{sym}_rsi.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - defensive
        return ""


def explain(config: str = "./config.yaml", outdir: str = "./reports", top_k: int = 10):
    """Generate a Markdown report explaining screened symbols."""
    cfg = load_cfg(config)
    symbols = cfg.get("symbols") or [cfg["futu"]["symbol"]]
    funds = load_fundamentals(
        os.path.join(os.path.dirname(config), "data", "fundamentals.csv")
        if os.path.exists(os.path.join(os.path.dirname(config), "data", "fundamentals.csv"))
        else "./data/fundamentals.csv"
    )

    sr_path = os.path.join(outdir, "screen_results.csv")
    if not os.path.exists(sr_path):
        print("No screen_results.csv. Run screener.py first.")
        return

    res = pd.read_csv(sr_path)
    passed = res[res["pass"] == True]
    agg = (
        passed.groupby("symbol").size().reset_index(name="score")
        .sort_values("score", ascending=False)
        .head(top_k)
    )

    lines = ["# Screener Explanation", ""]
    for _, row in agg.iterrows():
        sym = row["symbol"]
        df, err = fetch_kline(
            cfg["futu"]["host"],
            cfg["futu"]["port"],
            sym,
            cfg["futu"]["ktype"],
            cfg["futu"]["k_num"],
            cfg["futu"]["autype"],
        )
        ctx = build_context(df, fundamentals=funds, symbol=sym)
        lines.append(f"## {sym}  (rules passed: {int(row['score'])})")
        p1 = plot_price(df, sym, os.path.join(outdir, "images"))
        p2 = plot_rsi(df, sym, os.path.join(outdir, "images"))
        if p1:
            lines.append(f"![](images/{os.path.basename(p1)})")
        if p2:
            lines.append(f"![](images/{os.path.basename(p2)})")

        sub = passed[passed["symbol"] == sym]
        for _, r in sub.iterrows():
            rule = str(r["rule"])
            names = [n for n in names_required(rule) if n not in {"pct", "cross_up", "df_full"}]
            vals = {n: val_of(n, ctx) for n in names}
            lines.append(f"- **{r['id']} {r['name']}**: `{rule}`")
            if vals:
                pretty = ", ".join(
                    [
                        f"{k}={vals[k]:.4g}" if isinstance(vals[k], (int, float)) and vals[k] is not None else f"{k}={vals[k]}"
                        for k in vals
                    ]
                )
                lines.append(f"  - values: {pretty}")
        lines.append("")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "screener_explain.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved", os.path.join(outdir, "screener_explain.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--out", default="./reports")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    explain(args.config, args.out, args.top)
