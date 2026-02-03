import os
import pandas as pd
import requests

OUTPUT_FILE = "config/cse_symbols.csv"
os.makedirs("config", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def main():
    # This page contains a table with Symbol + Company Name
    url = "https://stockanalysis.com/list/colombo-stock-exchange/"

    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to load list page. HTTP {r.status_code}")

    # Pandas can read HTML tables directly
    tables = pd.read_html(r.text)

    if not tables:
        raise RuntimeError("No tables found on the page. Page layout may have changed.")

    # The first big table is the stock list
    df = tables[0].copy()

    # Expect columns like: No., Symbol, Company Name, ...
    if "Symbol" not in df.columns or "Company Name" not in df.columns:
        raise RuntimeError(f"Unexpected table columns: {list(df.columns)}")

    df = df[["Symbol", "Company Name"]].rename(columns={"Company Name": "company", "Symbol": "cse_symbol"})

    # Make simple base symbol like JKH from JKH.N0000
    df["symbol"] = df["cse_symbol"].astype(str).str.split(".").str[0].str.strip()

    # Yahoo mapping pattern often works: JKH.N0000 -> JKH-N0000.CM
    # (Some may still fail later, that's normal.)
    df["yahoo_ticker"] = df["cse_symbol"].astype(str).str.replace(".", "-", regex=False) + ".CM"

    # Clean
    df["company"] = df["company"].astype(str).str.strip()
    df["cse_symbol"] = df["cse_symbol"].astype(str).str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()

    # Remove blanks, duplicates
    df = df[df["cse_symbol"].str.len() > 0].drop_duplicates(subset=["cse_symbol"]).sort_values("cse_symbol")

    # Final column order
    df = df[["symbol", "company", "cse_symbol", "yahoo_ticker"]]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} companies to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
