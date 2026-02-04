import os
import pandas as pd
import urllib.request

OUT_FILE = "data/reference/company_master.csv"

SOURCES = [
    "https://stockanalysis.com/list/colombo-stock-exchange/",
    # Backup source (if you want later we can add more)
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def ensure_dirs():
    os.makedirs("data/reference", exist_ok=True)

def fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def parse_company_table_from_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found in HTML.")

    df = tables[0].copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "symbol" not in df.columns:
        raise RuntimeError(f"Missing 'symbol' column. Columns: {df.columns.tolist()}")

    name_col = None
    for c in df.columns:
        if c in {"company_name", "company"}:
            name_col = c
            break
    if not name_col:
        raise RuntimeError(f"Missing company name column. Columns: {df.columns.tolist()}")

    out = df[["symbol", name_col]].rename(columns={name_col: "company_name"})
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["company_name"] = out["company_name"].astype(str).str.strip()

    out = out[(out["symbol"] != "") & (out["company_name"] != "")]
    out = out.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)
    return out

def main():
    ensure_dirs()

    last_error = None
    for url in SOURCES:
        try:
            html = fetch_html(url)
            out = parse_company_table_from_html(html)
            out.to_csv(OUT_FILE, index=False)
            print(f"Saved: {OUT_FILE} ({len(out)} rows) from {url}")
            return
        except Exception as e:
            last_error = e
            print(f"Failed source: {url} -> {e}")

    raise RuntimeError(f"All sources failed. Last error: {last_error}")

if __name__ == "__main__":
    main()
