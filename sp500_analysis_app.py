import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Page config ---
st.set_page_config(layout="wide", page_title="Index Performance Analyzer")

# --- Load S&P500 data ---
@st.cache_data
def get_sp500_metadata():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={
        "Name": "Security",
        "Sector": "GICS Sector",
        "Industry": "GICS Sub-Industry"
    })
    cols = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    return df[cols]

# --- Load CSI300 data ---
@st.cache_data
def load_csi300_metadata():
    df = pd.read_excel("CSI 300.xlsx")

    def clean_ticker(raw):
        ticker = str(raw).strip()
        return ticker[:6] if ticker[:6].isdigit() else None

    df["Cleaned"] = df["Ticker"].apply(clean_ticker)
    df = df.dropna(subset=["Cleaned"])

    def fix_ticker(ticker):
        if ticker.startswith("6"):
            return ticker + ".SS"
        elif ticker.startswith("0") or ticker.startswith("3"):
            return ticker + ".SZ"
        else:
            return None

    df["Symbol"] = df["Cleaned"].apply(fix_ticker)
    df = df.dropna(subset=["Symbol"])

    return df[["Symbol", "Company", "Sector", "Industry Group"]].rename(
        columns={"Company": "Security", "Sector": "GICS Sector", "Industry Group": "GICS Sub-Industry"}
    )

# --- Download price data ---
@st.cache_data
def get_price_data(tickers, start_date, end_date):
    start_buffer = (pd.to_datetime(start_date) - timedelta(days=5)).strftime('%Y-%m-%d')
    end_buffer = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_buffer, end=end_buffer, group_by="ticker", auto_adjust=True, threads=True)

    price_data = {}
    for ticker in tickers:
        try:
            df = data[ticker][['Close', 'Volume']].copy()
            df['Daily % Change'] = df['Close'].pct_change() * 100
            df.dropna(inplace=True)
            df = df.loc[(df.index.date >= pd.to_datetime(start_date).date()) &
                        (df.index.date <= pd.to_datetime(end_date).date())]
            price_data[ticker] = df
        except Exception:
            continue
    return price_data

# --- Yuhhh ---
def compute_performance(price_data):
    perf = {}
    avg_volume = {}
    for ticker, df in price_data.items():
        perf[ticker] = df['Daily % Change'].sum()
        avg_volume[ticker] = df['Volume'].mean()
    return perf, avg_volume

def highlight_returns(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}; font-weight: bold'

def display_top_movers(performance, avg_volume, metadata, title, ascending=False):
    df = pd.DataFrame(performance.items(), columns=['Ticker', 'Return'])
    df['Avg Volume'] = df['Ticker'].map(avg_volume)
    df = df.merge(metadata, left_on='Ticker', right_on='Symbol', how='left')
    df = df[['Ticker', 'Security', 'Return', 'Avg Volume']].sort_values(by='Return', ascending=ascending).head(10)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    styled_df = df.style.format({'Return': '{:.2f}%', 'Avg Volume': '{:,.0f}'}).applymap(highlight_returns, subset=['Return'])
    st.subheader(title)
    st.dataframe(styled_df, use_container_width=True)

def display_group_performance(performance, avg_volume, metadata, group_col, title):
    df = pd.DataFrame(performance.items(), columns=['Ticker', 'Return'])
    df['Avg Volume'] = df['Ticker'].map(avg_volume)
    df = df.merge(metadata, left_on='Ticker', right_on='Symbol', how='left')
    group_perf = df.groupby(group_col).agg({
        'Return': 'mean',
        'Avg Volume': 'mean'
    }).sort_values(by='Return', ascending=False).round(2).reset_index()
    group_perf.rename(columns={'Return': 'Avg Return (%)', 'Avg Volume': 'Avg Volume'}, inplace=True)
    group_perf.index += 1
    st.subheader(title)
    st.dataframe(group_perf, use_container_width=True)

# --- Sidebar controls ---
st.sidebar.title("Index Selector")
index_choice = st.sidebar.selectbox("Choose Index", ["S&P 500", "CSI 300"])

st.sidebar.markdown("---")
today = datetime.today().date()
default_start = datetime(today.year, 1, 1).date()
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today, max_value=today)

if start_date > end_date:
    st.error("⚠️ Start date must be before end date.")
    st.stop()

# --- Load data & price ---
metadata = get_sp500_metadata() if index_choice == "S&P 500" else load_csi300_metadata()
tickers = metadata['Symbol'].dropna().unique().tolist()

with st.spinner("Downloading price data..."):
    price_data = get_price_data(tickers, start_date, end_date)

if not price_data:
    st.error("⚠️ No valid data returned. Try a wider or different date range.")
    st.stop()

performance, avg_volume = compute_performance(price_data)
latest_date = max(df.index.max() for df in price_data.values())

# --- Title ---
st.title(f"{index_choice} Performance Analyzer")
st.markdown(f"**Date Range:** `{start_date}` to `{end_date}`")
st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🏆 Top Movers", "📊 Group Performance", "🔍 Ticker Inspector"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        display_top_movers(performance, avg_volume, metadata, "Top 10 Gainers", ascending=False)
    with col2:
        display_top_movers(performance, avg_volume, metadata, "Top 10 Losers", ascending=True)

with tab2:
    display_group_performance(performance, avg_volume, metadata, "GICS Sector", "Sector Performance")
    st.markdown("___")
    display_group_performance(performance, avg_volume, metadata, "GICS Sub-Industry", "Industry Group Performance")

with tab3:
    st.sidebar.markdown("---")
    selected_ticker = st.sidebar.selectbox("Inspect Specific Ticker", ["None"] + sorted(price_data.keys()))
    if selected_ticker != "None":
        st.subheader(f"Cumulative % Return for `{selected_ticker}`")
        df = price_data[selected_ticker].copy().round(2)
        df['Cumulative % Change'] = df['Daily % Change'].cumsum()
        total_return = df['Daily % Change'].sum()

        st.line_chart(df['Cumulative % Change'])
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Total Movement:** `{total_return:.2f}%`")

# --- Footer ---
if price_data:
    try:
        latest_date = max(df.index.max() for df in price_data.values())
        latest_date_str = latest_date.strftime("%Y-%m-%d")
        st.markdown("---")
        st.caption(f"Data provided by yfinance • Last updated: {latest_date_str}")
    except Exception:
        st.markdown("---")
        st.caption("Data provided by yfinance • Last updated: Unknown")
