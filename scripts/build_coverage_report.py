import os
import pandas as pd
from datetime import datetime, timedelta

INDICATORS_FILE = "data/processed/indicators.csv"
MASTER_FILE = "data/reference/company_master.csv"
OUT_FILE = "data/processed/coverage_report.csv"

def main():
    os.makedirs("data/processed", exist_ok=True)

    ind = pd.read_csv(INDICATORS_FILE)
    master = pd.read_csv(MASTER_FILE)

    ind["date"] = pd.to_datetime(ind["date"], errors="coerce")
    ind["symbol"] = ind["symbol"].astype(str).str.strip()
    master["symbol"] = master["symbol"].astype(str).str.strip()

    today = ind["date"].max()

    summary = (
        ind.groupby("symbol")
        .agg(
            last_update=("date", "max"),
            days_of_data=("date", "count"),
            source=("source", "last"),
        )
        .reset_index()
    )

    summary["status"] = "OK"
    summary.loc[summary["last_update"] < today - timedelta(days=2), "status"] = "STALE"
    summary.loc[summary["days_of_data"] < 20, "status"] = "LOW_HISTORY"

    # Attach company names
    summary["symbol_key"] = summary["symbol"].str.split(".").str[0]
    master["symbol_key"] = master["symbol"].str.split(".").str[0]
    master_key = master.drop_duplicates("symbol_key")[["symbol_key", "company_name"]]

    summary = summary.merge(master_key, on="symbol_key", how="left")

    summary = summary[
        ["company_name", "symbol", "last_update", "days_of_data", "source", "status"]
    ].sort_values(["status", "company_name"])

    summary.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
