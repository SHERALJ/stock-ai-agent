import os
import pandas as pd
import requests
from datetime import datetime

SYMBOL_FILE = "config/cse_symbols.csv"
OUTPUT_FILE = "data/raw/daily_prices.csv"
FAILED_FILE = "data/raw/failed_prices.csv"
FAILED_HISTORY_FILE = "data/raw/failed_prices_history.csv"


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

    # If regularMarketPrice is missing, try previousClose
    if price is None:
        price = meta.get("previousClose")

    if price is None:
        raise RuntimeError(f"Missing regularMarketPrice for {yahoo_ticker}")

    return float(price)


def safe_read_existing(path: str) -> pd.DataFrame:
    cols = ["date", "symbol", "yahoo_ticker", "close_price", "source"]

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
    
def fetch_price_from_cse(cse_symbol: str) -> float:
    """
    Fallback: get price from CSE public endpoint used by their site.
    Note: endpoint format can change.
    """
    url = "https://www.cse.lk/api/companyInfoSummery"
    payload = {"symbol": cse_symbol}

    r = requests.post(url, data=payload, headers=HEADERS, timeout=20)

    if r.status_code != 200:
        raise RuntimeError(f"CSE HTTP {r.status_code} for {cse_symbol}")

    data = r.json()

    info = data.get("reqSymbolInfo") or {}
    price = info.get("lastTradedPrice")

    if price is None:
        raise RuntimeError(f"CSE missing lastTradedPrice for {cse_symbol}")

    return float(price)



def main():
    symbols_df = pd.read_csv(SYMBOL_FILE)
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    failed = []

    for _, row in symbols_df.iterrows():
        symbol = str(row["symbol"]).strip()
        cse_symbol = str(row.get("cse_symbol", "")).strip()

        cands = ticker_candidates(row)

        success = False
        last_error = "unknown error"

        # 1) Try Yahoo candidates
        for yahoo_ticker in cands:
            try:
                price = fetch_price(yahoo_ticker)
                rows.append({
                    "date": today,
                    "symbol": symbol,
                    "yahoo_ticker": yahoo_ticker,
                    "close_price": price,
                    "source": "yahoo"
                })
                print(f"Collected {symbol} ({yahoo_ticker}) from Yahoo: {price}")
                success = True
                break
            except Exception as e:
                last_error = str(e)

        # 2) If Yahoo failed, try CSE fallback once
        if not success and cse_symbol:
            try:
                price = fetch_price_from_cse(cse_symbol)
                rows.append({
                    "date": today,
                    "symbol": symbol,
                    "yahoo_ticker": "",
                    "close_price": price,
                    "source": "cse"
                })
                print(f"Collected {symbol} ({cse_symbol}) from CSE: {price}")
                success = True
            except Exception as e:
                last_error = str(e)

        # 3) If still failed, record failure with a real error
        if not success:
            failed.append({
                "date": today,
                "symbol": symbol,
                "yahoo_ticker": cands[0] if cands else "",
                "error": (last_error or "unknown error")[:250]
            })
            print(f"Failed {symbol} (Yahoo + CSE): {last_error}")





    df_new = pd.DataFrame(rows, columns=["date", "symbol", "yahoo_ticker", "close_price", "source"])
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


    df_failed_new = pd.DataFrame(
        failed,
        columns=["date", "symbol", "yahoo_ticker", "error"]
    )

    if os.path.exists(FAILED_FILE):
        df_failed_old = pd.read_csv(FAILED_FILE)
        df_failed_all = pd.concat([df_failed_old, df_failed_new], ignore_index=True)
    else:
        df_failed_all = df_failed_new

    # Keep only latest failure per symbol per date
    df_failed_all = df_failed_all.drop_duplicates(
        subset=["date", "symbol"],
        keep="last"
    )

    df_failed_all.to_csv(FAILED_FILE, index=False)
    print(f"Failed list saved to {FAILED_FILE} ({len(df_failed_all)} rows)")

    # Save today's failures (overwrite)
    df_failed = pd.DataFrame(failed, columns=["date", "symbol", "yahoo_ticker", "error"])
    df_failed.to_csv(FAILED_FILE, index=False)
    print(f"Failed list saved to {FAILED_FILE} ({len(df_failed)} rows)")

    # Save failure history (append + de-duplicate)
    if os.path.exists(FAILED_HISTORY_FILE):
        df_hist_old = pd.read_csv(FAILED_HISTORY_FILE)
        df_hist_all = pd.concat([df_hist_old, df_failed], ignore_index=True)
    else:
        df_hist_all = df_failed

    df_hist_all = df_hist_all.drop_duplicates(subset=["date", "symbol", "error"], keep="last")
    df_hist_all.to_csv(FAILED_HISTORY_FILE, index=False)
    print(f"Failed history saved to {FAILED_HISTORY_FILE} ({len(df_hist_all)} rows)")



if __name__ == "__main__":
    main()
