import os
import pandas as pd
import numpy as np

RAW_FILE = "data/raw/daily_prices.csv"
OUT_ALL = "data/processed/indicators.csv"
OUT_LATEST = "data/processed/latest_signals.csv"

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)

def main():
    ensure_dirs()
    df = pd.read_csv(RAW_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # Map your schema -> standard schema
    if "close" not in df.columns:
        if "close_price" in df.columns:
            df["close"] = df["close_price"]
        else:
            raise ValueError(f"Missing close/close_price. Columns: {df.columns.tolist()}")

    # (Optional) keep only what we need + any OHLCV you may add later
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns: {df.columns.tolist()}")


    def per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        close = g["close"]

        g["ret_1d"] = close.pct_change()
        g["sma_50"] = close.rolling(50).mean()
        g["sma_200"] = close.rolling(200).mean()
        g["ema_20"] = close.ewm(span=20, adjust=False).mean()

        g["rsi_14"] = rsi(close, 14)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        if {"high", "low"}.issubset(g.columns):
            g["atr_14"] = atr(g, 14)
        else:
            g["atr_14"] = np.nan

        g["vol_20"] = g["ret_1d"].rolling(20).std() * np.sqrt(252)

        g["high_252"] = close.rolling(252).max()
        g["dist_to_52w_high"] = (close / g["high_252"]) - 1

        g["peak_close"] = close.cummax()
        g["drawdown"] = (close / g["peak_close"]) - 1

        if "volume" in g.columns:
            g["vol_sma_20"] = g["volume"].rolling(20).mean()
            g["vol_spike"] = g["volume"] / g["vol_sma_20"]
        else:
            g["vol_sma_20"] = np.nan
            g["vol_spike"] = np.nan

        # Simple long-term plus short-term insight labels
        g["trend_long"] = np.where(g["sma_50"] > g["sma_200"], "UP", "DOWN")
        g["rsi_state"] = np.where(g["rsi_14"] >= 70, "OVERBOUGHT",
                          np.where(g["rsi_14"] <= 30, "OVERSOLD", "NORMAL"))

        return g

    out = df.groupby("symbol", group_keys=False).apply(per_symbol)

    out.to_csv(OUT_ALL, index=False)

    latest = out.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    latest = latest.sort_values("symbol")
    latest.to_csv(OUT_LATEST, index=False)

    print(f"Saved: {OUT_ALL}")
    print(f"Saved: {OUT_LATEST}")

if __name__ == "__main__":
    main()
