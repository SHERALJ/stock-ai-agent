import os
import pandas as pd
import numpy as np

LATEST_DAILY_FILE = "data/processed/latest_signals.csv"
INDICATORS_WEEKLY_FILE = "data/processed/indicators_weekly.csv"
MASTER_FILE = "data/reference/company_master.csv"
OUT_FILE = "data/processed/watchlist_top20.csv"


def ensure_dirs():
    os.makedirs("data/processed", exist_ok=True)


def add_company_names(df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    master = master.copy()
    master["symbol"] = master["symbol"].astype(str).str.strip()

    df["symbol_key"] = df["symbol"].str.split(".").str[0]
    master["symbol_key"] = master["symbol"].str.split(".").str[0]
    master_key = master.drop_duplicates("symbol_key")[["symbol_key", "company_name"]]

    return df.merge(master_key, on="symbol_key", how="left")


def main():
    ensure_dirs()

    if not os.path.exists(LATEST_DAILY_FILE):
        raise FileNotFoundError(f"Missing: {LATEST_DAILY_FILE}. Run build_indicators.py first.")
    if not os.path.exists(INDICATORS_WEEKLY_FILE):
        raise FileNotFoundError(f"Missing: {INDICATORS_WEEKLY_FILE}. Run build_indicators.py first.")
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError(f"Missing: {MASTER_FILE}. Run update_company_master.py first.")

    daily_latest = pd.read_csv(LATEST_DAILY_FILE)
    weekly = pd.read_csv(INDICATORS_WEEKLY_FILE)
    master = pd.read_csv(MASTER_FILE)

    if "date" in daily_latest.columns:
        daily_latest["date"] = pd.to_datetime(daily_latest["date"], errors="coerce")
    if "date" in weekly.columns:
        weekly["date"] = pd.to_datetime(weekly["date"], errors="coerce")

    # Latest weekly row per symbol (best for long-term trend)
    weekly["symbol"] = weekly["symbol"].astype(str).str.strip()
    weekly_latest = (
        weekly.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # Join daily + weekly on symbol
    daily_latest["symbol"] = daily_latest["symbol"].astype(str).str.strip()
    merged = daily_latest.merge(
        weekly_latest[["symbol", "sma_50", "sma_200", "drawdown", "dist_to_52w_high", "vol_20"]],
        on="symbol",
        how="left",
        suffixes=("", "_weekly"),
    )

    merged = add_company_names(merged, master)

    # Score (weekly trend + daily timing)
    score = pd.Series(0, index=merged.index, dtype="int")

    # Weekly long-term trend (strong weight)
    score += (merged["sma_50"] > merged["sma_200"]).fillna(False).astype(int) * 3

    # Daily close above daily SMA50 (medium)
    if "close" in merged.columns and "sma_50" in merged.columns:
        score += (merged["close"] > merged["sma_50"]).fillna(False).astype(int) * 1

    # RSI “healthy zone” (timing)
    if "rsi_14" in merged.columns:
        score += ((merged["rsi_14"] >= 40) & (merged["rsi_14"] <= 65)).fillna(False).astype(int) * 1

    # Not too far from 52-week high (strength)
    if "dist_to_52w_high" in merged.columns:
        score += (merged["dist_to_52w_high"] > -0.25).fillna(False).astype(int) * 1

    # Avoid deep drawdowns (risk)
    if "drawdown" in merged.columns:
        score -= (merged["drawdown"] < -0.35).fillna(False).astype(int) * 1

    # MACD histogram positive (short-term momentum)
    if "macd_hist" in merged.columns:
        score += (merged["macd_hist"] > 0).fillna(False).astype(int) * 1

    merged["score"] = score

    # Output columns
    cols = [
        "company_name",
        "symbol",
        "date",
        "close",
        "score",
        "trend_long",
        "rsi_14",
        "rsi_state",
        "macd_hist",
        "dist_to_52w_high",
        "drawdown",
        "vol_20",
    ]
    cols = [c for c in cols if c in merged.columns]
    out = merged[cols].copy()

    # Format
    for c in ["dist_to_52w_high", "drawdown", "vol_20"]:
        if c in out.columns:
            out[c] = (out[c] * 100).round(2)
    for c in ["rsi_14", "macd_hist", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out = out.sort_values(["score", "company_name"], ascending=[False, True]).head(20)
    out.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
