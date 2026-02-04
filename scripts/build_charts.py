import os
import pandas as pd
import plotly.graph_objects as go

IN_FILE = "data/processed/indicators.csv"
OUT_DIR = "reports/charts"

def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def build_symbol_chart(g: pd.DataFrame, symbol: str):
    g = g.sort_values("date")

    fig = go.Figure()

    if {"open","high","low","close"}.issubset(g.columns):
        fig.add_trace(go.Candlestick(
            x=g["date"],
            open=g["open"], high=g["high"], low=g["low"], close=g["close"],
            name="Price"
        ))
    else:
        fig.add_trace(go.Scatter(x=g["date"], y=g["close"], name="Close"))

    # Overlays
    for col in ["ema_20", "sma_50", "sma_200"]:
        if col in g.columns:
            fig.add_trace(go.Scatter(x=g["date"], y=g[col], name=col))

    fig.update_layout(
        title=f"{symbol} Price with EMA20, SMA50, SMA200",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    out_path = os.path.join(OUT_DIR, f"{symbol}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")

def main():
    ensure_dirs()
    df = pd.read_csv(IN_FILE)
    df["date"] = pd.to_datetime(df["date"])

    for symbol, g in df.groupby("symbol"):
        build_symbol_chart(g, symbol)

    print(f"Charts saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
