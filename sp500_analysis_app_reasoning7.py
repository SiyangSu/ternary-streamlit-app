import os
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests

# --- Page config ---
st.set_page_config(layout="wide", page_title="Index Performance Analyzer")

# --- load API key ---
api_key = os.getenv("API_KEY")

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
    df = pd.read_excel("CSI 300.xlsx")  # Replace with your file

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

# --- Compute performance ---
def compute_performance(price_data):
    perf = {}
    avg_volume = {}
    for ticker, df in price_data.items():
        perf[ticker] = df['Daily % Change'].sum()
        avg_volume[ticker] = df['Volume'].mean()
    return perf, avg_volume

# --- Highlight returns ---
def highlight_returns(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}; font-weight: bold'

# --- Mistral AI API ---
MISTRAL_API_KEY = api_key  #Replace with your API key

def get_ai_reasoning(company_name, price_change, start_date, end_date, index_choice="S&P 500"):
    """Call Mistral AI to generate reasoning for why the company moved, with Chinese translation for CSI300."""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Base English prompt
    prompt_en = (
        f"Explain in 4 sentences or less why {company_name} stock moved {price_change:.2f}% between {start_date} and {end_date}. "
        "Only include specific news, events, or announcements that contributed to the rise or fall. "
        "Include specific information like company name and dates."
    )

    if index_choice == "CSI 300":
        # Wrap with translation instructions for China-specific search
        user_prompt = (
            f"Translate the following prompt to Chinese, fetch relevant Chinese news or announcements for the stock, "
            f"and provide the reasoning in Chinese:\n{prompt_en}\n\n"
            "Then translate the Chinese reasoning back to English for output. "
            "Output only the english reasoning, omit all intermediate chinese. "
        )
    else:
        user_prompt = prompt_en

    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are a financial analyst explaining stock movements."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error fetching reasoning: {e}"

# --- Display top/bottom movers with AI reasoning ---
def display_top_movers_with_ai(performance, avg_volume, metadata, title, start_date, end_date, index_choice, ascending=False):
    # Prepare top/bottom 10 dataframe
    df = pd.DataFrame(performance.items(), columns=['Ticker', 'Return'])
    df['Avg Volume'] = df['Ticker'].map(avg_volume)
    df = df.merge(metadata, left_on='Ticker', right_on='Symbol', how='left')
    # Include Industry column
    df = df[['Ticker', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'Return', 'Avg Volume']].sort_values(by='Return', ascending=ascending).head(10)
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    # Display table with green/red formatting
    st.subheader(title)
    st.dataframe(
        df.style.format({'Return':'{:.2f}%', 'Avg Volume':'{:,.0f}'}).applymap(highlight_returns, subset=['Return']),
        use_container_width=True
    )

    # Display AI reasoning for each company
    for i, row in df.iterrows():
        ticker = row['Ticker']
        company = row['Security']
        industry = row['GICS Sub-Industry']
        change = row['Return']
        reasoning = get_ai_reasoning(company, change, start_date, end_date, index_choice=index_choice)
        st.markdown(f"**AI reasoning for {ticker} ({company}, {industry}):** {reasoning}")
        st.markdown("---")

    # --- Aggregated summary for all top/bottom companies ---
    top_bottom_summary = "Summarize the overall performance of these companies:\n\n"
    for _, row in df.iterrows():
        top_bottom_summary += f"{row['Ticker']} ({row['Security']}, {row['GICS Sub-Industry']}): {row['Return']:.2f}%\n"
    top_bottom_summary += (f"\nExplain overall trends of these 10 companies, highlighting why particular industries "
                           f"performed well or poorly between {start_date} and {end_date}.")

    summary_reasoning = get_ai_reasoning("Top/Bottom companies summary", 0, start_date, end_date, index_choice=index_choice)
    st.markdown("### Overall Performance Summary")
    st.markdown(summary_reasoning)

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

# --- Title ---
st.title(f"{index_choice} Performance Analyzer")
st.markdown(f"**Date Range:** `{start_date}` to `{end_date}`")
st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🏆 Top Movers", "📊 Group Performance", "🔍 Ticker Inspector"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        display_top_movers_with_ai(performance, avg_volume, metadata, "Top 10 Gainers", start_date, end_date, index_choice=index_choice, ascending=False)
    with col2:
        display_top_movers_with_ai(performance, avg_volume, metadata, "Top 10 Losers", start_date, end_date, index_choice=index_choice, ascending=True)

with tab2:
    # Optional: show sector performance
    df_perf = pd.DataFrame(performance.items(), columns=['Ticker', 'Return'])
    df_perf['Avg Volume'] = df_perf['Ticker'].map(avg_volume)
    df_perf = df_perf.merge(metadata, left_on='Ticker', right_on='Symbol', how='left')
    group_perf = df_perf.groupby('GICS Sector').agg({'Return':'mean', 'Avg Volume':'mean'}).round(2)
    st.subheader("Sector Performance")
    st.dataframe(group_perf)

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
