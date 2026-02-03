import os
import pandas as pd
import requests
from datetime import datetime

SYMBOL_FILE = "config/cse_symbols.csv"
OUTPUT_FILE = "data/raw/daily_prices.csv"
FAILED_FILE = "data/raw/failed_prices.csv"

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

def ticker_candidates(row: pd.Series) -> list[str]:
    """
    Return a list of possible Yahoo tickers to try, ordered from best guess to worst.
    """
    cands = []

    # 1) If user provided yahoo_ticker in CSV, try it first
    if "yahoo_ticker" in row and isinstance(row["yahoo_ticker"], str) and row["yahoo_ticker"].strip():
        cands.append(row["yahoo_ticker"].strip())

    # 2) From cse_symbol (JKH.N0000 -> JKH-N0000.CM)
    cse_symbol = ""
    if "cse_symbol" in row and isinstance(row["cse_symbol"], str) and row["cse_symbol"].strip():
        cse_symbol = row["cse_symbol"].strip()
        cands.append(cse_symbol.replace(".", "-") + ".CM")

        # If it's a preference share like AAF.P0000, also try normal share AAF.N0000
        parts = cse_symbol.split(".")
        if len(parts) == 2:
            base, series = parts[0], parts[1]
            if series.upper().startswith("P"):
                cands.append(f"{base}-N0000.CM")

    # 3) Base symbol fallbacks
    base_symbol = str(row["symbol"]).strip()
    cands.append(f"{base_symbol}-N0000.CM")   # common pattern
    cands.append(f"{base_symbol}.CM")         # sometimes used
    cands.append(base_symbol)                 # last try

    # Remove duplicates while keeping order
    seen = set()
    unique = []
    for t in cands:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


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
    failed = []

    for _, row in symbols_df.iterrows():
        symbol = str(row["symbol"]).strip()

        cands = ticker_candidates(row)

        success = False
        last_error = ""

        for yahoo_ticker in cands:
            try:
                price = fetch_price(yahoo_ticker)

                rows.append({
                    "date": today,
                    "symbol": symbol,
                    "yahoo_ticker": yahoo_ticker,
                    "close_price": price
                })

                print(f"Collected {symbol} ({yahoo_ticker}): {price}")
                success = True
                break

            except Exception as e:
                last_error = str(e)

        if not success:
            print(f"Failed {symbol} (tried {len(cands)} tickers): {last_error}")

            failed.append({
                "date": today,
                "symbol": symbol,
                "yahoo_ticker": cands[0] if cands else "",
                "error": last_error[:250]
            })



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

    df_failed = pd.DataFrame(failed, columns=["date", "symbol", "yahoo_ticker", "error"])
    df_failed.to_csv(FAILED_FILE, index=False)
    print(f"Failed list saved to {FAILED_FILE} ({len(df_failed)} rows)")


if __name__ == "__main__":
    main()
