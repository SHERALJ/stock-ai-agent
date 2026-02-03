import os
import pandas as pd
import requests
from datetime import datetime

SYMBOL_FILE = "config/cse_symbols.csv"
OUTPUT_FILE = "data/raw/daily_prices.csv"

os.makedirs("data/raw", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

def to_yahoo_ticker(row: pd.Series) -> str:
    if "yahoo_ticker" in row and isinstance(row["yahoo_ticker"], str) and row["yahoo_ticker"].strip():
        return row["yahoo_ticker"].strip()

    if "cse_symbol" in row and isinstance(row["cse_symbol"], str) and row["cse_symbol"].strip():
        return row["cse_symbol"].strip().replace(".", "-") + ".CM"

    return str(row["symbol"]).strip() + ".CM"

def fetch_price(yahoo_ticker: str) -> float:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}?interval=1d&range=5d"
    r = requests.get(url, headers=HEADERS, timeout=15)

    if r.status_code != 200:
        snippet = r.text[:120].replace("\n", " ")
        raise RuntimeError(f"HTTP {r.status_code} for {yahoo_ticker}. Body starts: {snippet}")

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "json" not in ctype:
        snippet = r.text[:120].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response for {yahoo_ticker}. Content-Type={ctype}. Body starts: {snippet}")

    data = r.json()
    result = data.get("chart", {}).get("result")

    if not result:
        err = data.get("chart", {}).get("error")
        raise RuntimeError(f"No result for {yahoo_ticker}. Error={err}")

    meta = result[0].get("meta", {})
    price = meta.get("regularMarketPrice")

    if price is None:
        raise RuntimeError(f"Missing regularMarketPrice for {yahoo_ticker}")

    return float(price)

def safe_read_existing(path: str) -> pd.DataFrame:
    cols = ["date", "symbol", "yahoo_ticker", "close_price"]

    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)

    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return pd.DataFrame(columns=cols)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

def main():
    symbols_df = pd.read_csv(SYMBOL_FILE)
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []

    for _, row in symbols_df.iterrows():
        symbol = str(row["symbol"]).strip()
        yahoo_ticker = to_yahoo_ticker(row)

        try:
            price = fetch_price(yahoo_ticker)

            rows.append({
                "date": today,
                "symbol": symbol,
                "yahoo_ticker": yahoo_ticker,
                "close_price": price
            })

            print(f"Collected {symbol} ({yahoo_ticker}): {price}")

        except Exception as e:
            print(f"Failed {symbol} ({yahoo_ticker}): {e}")

    df_new = pd.DataFrame(rows, columns=["date", "symbol", "yahoo_ticker", "close_price"])
    df_old = safe_read_existing(OUTPUT_FILE)

    if df_old.empty:
        df_all = df_new.copy()
    else:
        df_all = pd.concat([df_old, df_new], ignore_index=True)

    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["date", "symbol"], keep="last")

    df_all.to_csv(OUTPUT_FILE, index=False)
    print("Daily price collection completed.")

if __name__ == "__main__":
    main()
