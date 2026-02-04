import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

INDICATORS_FILE = BASE_DIR / "data" / "processed" / "indicators.csv"
INDICATORS_WEEKLY_FILE = BASE_DIR / "data" / "processed" / "indicators_weekly.csv"
LATEST_FILE = BASE_DIR / "data" / "processed" / "latest_signals.csv"
MASTER_FILE = BASE_DIR / "data" / "reference" / "company_master.csv"
WATCHLIST_FILE = BASE_DIR / "data" / "processed" / "watchlist_top20.csv"
COVERAGE_FILE = BASE_DIR / "data" / "processed" / "coverage_report.csv"

st.subheader("Data Coverage & Health")
if COVERAGE_FILE.exists():
    cov = pd.read_csv(COVERAGE_FILE)
    st.dataframe(cov, use_container_width=True, height=400)
else:
    st.info("Run: python scripts/build_coverage_report.py")


st.set_page_config(page_title="CSE Compare Dashboard", layout="wide")


@st.cache_data
def load_data(indicators_path: Path):
    ind = pd.read_csv(indicators_path)
    ind["date"] = pd.to_datetime(ind["date"])

    latest = pd.read_csv(LATEST_FILE)
    if "date" in latest.columns:
        latest["date"] = pd.to_datetime(latest["date"])

    master = pd.read_csv(MASTER_FILE)

    # Clean strings
    ind["symbol"] = ind["symbol"].astype(str).str.strip()
    latest["symbol"] = latest["symbol"].astype(str).str.strip()
    master["symbol"] = master["symbol"].astype(str).str.strip()

    # Build a common key for mapping names (handles JKH.N0000 vs JKH)
    ind["symbol_key"] = ind["symbol"].str.split(".").str[0]
    latest["symbol_key"] = latest["symbol"].str.split(".").str[0]
    master["symbol_key"] = master["symbol"].str.split(".").str[0]

    # Keep one row per key
    master_key = master.drop_duplicates("symbol_key")[["symbol_key", "company_name"]].copy()

    # Attach company names
    ind = ind.merge(master_key, on="symbol_key", how="left")
    latest = latest.merge(master_key, on="symbol_key", how="left")

    # Friendly label
    ind["label"] = ind["company_name"].fillna("Unknown") + " (" + ind["symbol"] + ")"
    latest["label"] = latest["company_name"].fillna("Unknown") + " (" + latest["symbol"] + ")"

    return ind, latest


def normalize_price(g: pd.DataFrame, price_col="close") -> pd.Series:
    base = g[price_col].iloc[0]
    if pd.isna(base) or base == 0:
        return g[price_col] * np.nan
    return (g[price_col] / base) * 100.0


def make_chart(df: pd.DataFrame, view: str, show_ma: bool):
    fig = go.Figure()

    for label, g in df.groupby("label"):
        g = g.sort_values("date")

        if view == "Normalized Price (start=100)":
            y = normalize_price(g, "close")
            fig.add_trace(go.Scatter(x=g["date"], y=y, name=label))
            y_title = "Normalized (100 = start)"

        elif view == "Close Price":
            fig.add_trace(go.Scatter(x=g["date"], y=g["close"], name=label))
            y_title = "Price"

        elif view == "Drawdown":
            fig.add_trace(go.Scatter(x=g["date"], y=g["drawdown"] * 100, name=label))
            y_title = "Drawdown (%)"

        elif view == "RSI (14)":
            fig.add_trace(go.Scatter(x=g["date"], y=g["rsi_14"], name=label))
            y_title = "RSI"

        elif view == "MACD Histogram":
            fig.add_trace(go.Scatter(x=g["date"], y=g["macd_hist"], name=label))
            y_title = "MACD Hist"

        elif view == "Volatility (20D annualized)":
            fig.add_trace(go.Scatter(x=g["date"], y=g["vol_20"] * 100, name=label))
            y_title = "Volatility (%)"

        else:
            fig.add_trace(go.Scatter(x=g["date"], y=g["close"], name=label))
            y_title = "Price"

        # Optional moving averages (only meaningful on Close Price view)
        if show_ma and view == "Close Price":
            for ma in ["sma_50", "sma_200"]:
                if ma in g.columns and g[ma].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=g["date"],
                            y=g[ma],
                            name=f"{label} {ma}",
                            line=dict(dash="dot"),
                        )
                    )

    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=y_title)
    return fig


def signals_table(latest: pd.DataFrame, selected_symbols: list[str]) -> pd.DataFrame:
    view = latest[latest["symbol"].isin(selected_symbols)].copy()

    cols = [
        "company_name",
        "symbol",
        "date",
        "close",
        "trend_long",
        "rsi_14",
        "rsi_state",
        "dist_to_52w_high",
        "drawdown",
        "vol_20",
        "score",
    ]
    cols = [c for c in cols if c in view.columns]
    view = view[cols].copy()

    if "dist_to_52w_high" in view.columns:
        view["dist_to_52w_high"] = (view["dist_to_52w_high"] * 100).round(2)
    if "drawdown" in view.columns:
        view["drawdown"] = (view["drawdown"] * 100).round(2)
    if "vol_20" in view.columns:
        view["vol_20"] = (view["vol_20"] * 100).round(2)
    if "rsi_14" in view.columns:
        view["rsi_14"] = view["rsi_14"].round(2)
    if "close" in view.columns:
        view["close"] = view["close"].round(2)

    sort_cols = [c for c in ["score", "trend_long", "company_name"] if c in view.columns]
    if sort_cols:
        # score desc, trend_long desc, name asc (as available)
        ascending = []
        for c in sort_cols:
            ascending.append(False if c in {"score", "trend_long"} else True)
        view = view.sort_values(sort_cols, ascending=ascending)

    return view


def main():
    st.title("Sri Lanka (CSE) Stocks: Compare + Signals")

    # Sidebar controls MUST come before timeframe is used
    st.sidebar.header("Controls")

    timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=0)

    view = st.sidebar.selectbox(
        "Chart view",
        [
            "Normalized Price (start=100)",
            "Close Price",
            "Drawdown",
            "RSI (14)",
            "MACD Histogram",
            "Volatility (20D annualized)",
        ],
        index=0,
    )

    show_ma = st.sidebar.checkbox("Show SMA50 & SMA200 (Close Price view only)", value=False)

    # Debug
    st.subheader("Debug")
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("Daily indicators file:", str(INDICATORS_FILE), "exists:", INDICATORS_FILE.exists())
    st.write("Weekly indicators file:", str(INDICATORS_WEEKLY_FILE), "exists:", INDICATORS_WEEKLY_FILE.exists())
    st.write("Latest file:", str(LATEST_FILE), "exists:", LATEST_FILE.exists())
    st.write("Master file:", str(MASTER_FILE), "exists:", MASTER_FILE.exists())
    st.write("Watchlist file:", str(WATCHLIST_FILE), "exists:", WATCHLIST_FILE.exists())

    # Guardrails
    for path in [LATEST_FILE, MASTER_FILE]:
        if not path.exists():
            st.error(f"Missing file: {path}")
            st.stop()

    ind_path = INDICATORS_FILE if timeframe == "Daily" else INDICATORS_WEEKLY_FILE
    if not ind_path.exists():
        st.error(f"Missing file: {ind_path}. Run scripts/build_indicators.py")
        st.stop()

    ind, latest = load_data(ind_path)

    # Watchlist table (optional)
    st.subheader("Top 20 Watchlist (score)")
    if WATCHLIST_FILE.exists():
        wl = pd.read_csv(WATCHLIST_FILE)
        st.dataframe(wl, use_container_width=True, height=420)
    else:
        st.info("Watchlist not found. Run: python scripts/build_watchlist.py")

    # Build dropdown options from symbols that actually exist in indicators
    available = ind[["symbol", "label"]].drop_duplicates().sort_values("label")
    options = available["label"].tolist()
    label_to_symbol = dict(zip(available["label"], available["symbol"]))

    if not options:
        st.error("No symbols available in indicators file. Run scripts/build_indicators.py")
        st.stop()

    default_pick = options[:5] if len(options) >= 5 else options
    picked_labels = st.sidebar.multiselect("Select companies", options=options, default=default_pick)
    picked_symbols = [label_to_symbol[lbl] for lbl in picked_labels]

    # Date range
    min_date = ind["date"].min()
    max_date = ind["date"].max()
    start, end = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    # Filter data
    dff = ind[ind["symbol"].isin(picked_symbols)].copy()
    dff = dff[(dff["date"] >= pd.to_datetime(start)) & (dff["date"] <= pd.to_datetime(end))]

    if dff.empty:
        st.warning("No data for selected companies in the selected date range.")
        st.stop()

    # Main layout
    left, right = st.columns([2, 1], gap="large")

    with left:
        fig = make_chart(dff, view=view, show_ma=show_ma)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Latest signals")
        tbl = signals_table(latest, picked_symbols)
        st.dataframe(tbl, use_container_width=True, height=520)
        st.caption("Tip: Normalized price is best for comparing performance. Drawdown shows risk.")

    st.divider()

    # Single company detail view
    st.subheader("Open a single company detail view")
    one_label = st.selectbox("Choose one company", options=options, index=0)
    one_symbol = label_to_symbol[one_label]

    one = ind[ind["symbol"] == one_symbol].sort_values("date")
    one = one[(one["date"] >= pd.to_datetime(start)) & (one["date"] <= pd.to_datetime(end))]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=one["date"], y=one["close"], name="Close"))

    for ma in ["ema_20", "sma_50", "sma_200"]:
        if ma in one.columns and one[ma].notna().any():
            fig2.add_trace(go.Scatter(x=one["date"], y=one[ma], name=ma))

    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
    fig2.update_xaxes(title="Date")
    fig2.update_yaxes(title="Price")
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
