
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Top 3 Ticker Forecasts", layout="wide")

st.title("📊 Top 3 Ticker Forecast Summary")

# Load CSV files
data_dir = "historical_price_data_top3"
tickers = [f.replace("_price_history.csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")]

# Simulated: load forecasted values and news headlines
# You may replace these with actual DataFrame reads or API outputs
@st.cache_data
def load_forecast_and_news(ticker):
    df = pd.read_csv(f"{data_dir}/{ticker}_price_history.csv")
    forecast = df.tail(5)  # Simulating last 5 days as forecast
    news = f"Latest headline related to {ticker.upper()} goes here."
    return df, forecast, news

for ticker in tickers[:3]:
    st.subheader(f"📈 Ticker: {ticker.upper()}")

    df, forecast, news = load_forecast_and_news(ticker)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Latest Forecasted Values:**")
        st.dataframe(forecast[["Date", "Price"]].rename(columns={"Price": "Forecasted Price"}))

        st.markdown("**📰 Related News:**")
        st.info(news)

    with col2:
        st.markdown("**📉 Price Trend Plot:**")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Price"], label="Historical Price", marker="o")
        ax.set_title(f"{ticker.upper()} - Historical Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)


# Necessary libraries
#!pip install yfinance
#!pip install newspaper3k
#!pip install transformers torch
#!pip install yfinance
#!pip install chronos-ts --upgrade --quiet
import json

# Code to retrieve yesterday news from NewSAPI.

import requests
import pandas as pd
from datetime import datetime, timedelta

# 🔑 Enter your NewsAPI key here
NEWSAPI_KEY = "c32779e494d04276b24ac0eb577c5ca2"

def fetch_yesterdays_news():
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")

    query = (
    "stocks OR stock OR market OR earnings OR inflation OR layoffs OR fed OR economic data "
    "OR acquisition OR merger OR buyout OR billion OR million OR IPO OR funding "
    "OR forecast OR guidance OR quarterly results OR revenue OR profits OR shares "
    "OR dividends OR buybacks OR takeover OR analysts OR downgrade OR upgrade"
    )

    domains = (
        "bloomberg.com,cnn.com,cnbc.com,wsj.com,reuters.com,marketwatch.com,"
        "yahoo.com,investopedia.com,seekingalpha.com,fool.com,fortune.com,"
        "forbes.com,techcrunch.com,businessinsider.com,barrons.com"
    )

    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={date_str}&to={date_str}"
        f"&language=en&sortBy=publishedAt"
        f"&pageSize=100"
        f"&domains={domains}"
        f"&apiKey={NEWSAPI_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    if "articles" in data:
        articles = data["articles"]
        df = pd.DataFrame([{
            "title": article["title"],
            "description": article["description"],
            "publishedAt": article["publishedAt"],
            "source": article["source"]["name"]
        } for article in articles])
        return df
    else:
        print("No articles found or error in API call.")
        return pd.DataFrame()

# 🚀 Run it
news_df = fetch_yesterdays_news()
news_df.to_csv("news_headlines.csv", index=False)
news_df.head(10)  # Preview the headlines



import spacy
nlp = spacy.load("en_core_web_sm")


#Loading company names and tickers
import pandas as pd

def load_sp500_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)

    print("📊 Loaded columns:", df.columns.tolist())  # Debugging

    # Fix column names if needed
    if 'Name' not in df.columns or 'Symbol' not in df.columns:
        if len(df.columns) >= 2:
            df.columns = ['Symbol', 'Name'] + list(df.columns[2:])
        else:
            raise ValueError("CSV does not have expected columns.")

    return {row['Name'].lower(): row['Symbol'] for _, row in df.iterrows()}



# ✅ Named Entity Recognition + Ticker Extraction
def extract_companies_from_articles(news_df, known_companies):
    """
    Extracts company mentions from a DataFrame of news articles and maps them to S&P 500 tickers.

    Args:
        news_df (DataFrame): News articles with 'title' and 'description' columns
        known_companies (dict): Mapping of company names (lowercase) to tickers

    Returns:
        List of matched tickers
    """
    mentioned_tickers = set()
    articles = news_df.to_dict(orient="records")  # ✅ Ensure correct format

    for article in articles:
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "ORG":
                company_name = ent.text.lower()
                for known_name, ticker in known_companies.items():
                    if company_name in known_name:  # simple fuzzy match
                        mentioned_tickers.add(ticker)

    return list(mentioned_tickers)

news_df = fetch_yesterdays_news()
known_companies = load_sp500_tickers()

mentioned_tickers = extract_companies_from_articles(news_df, known_companies)
print("🧠 Tickers mentioned in yesterday’s news:", mentioned_tickers)


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from collections import defaultdict

# ✅ Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ✅ Get sentiment for a piece of text
def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    sentiment_idx = np.argmax(probs)
    sentiment_label = ["negative", "neutral", "positive"][sentiment_idx]
    score = probs[sentiment_idx]
    return sentiment_label, float(score)

# ✅ Score sentiment ONLY for tickers from Step 2 (your extracted tickers)
def score_sentiment_for_mentioned_tickers(news_df, known_companies, mentioned_tickers):
    from collections import defaultdict
    import numpy as np
    import pandas as pd

    # 🔄 Reverse map tickers -> company names
    ticker_to_name = {
        ticker: name
        for name, ticker in known_companies.items()
        if ticker in mentioned_tickers
    }

    sentiment_records = defaultdict(list)

    for _, article in news_df.iterrows():
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        sentiment, score = get_finbert_sentiment(text)

        for ticker in mentioned_tickers:
            company_name = ticker_to_name.get(ticker, "").lower()
            if ticker.lower() in text or company_name in text:
                sentiment_records[ticker].append((sentiment, score))

    # 📊 Aggregate results
    results = []
    for ticker, records in sentiment_records.items():
        sentiments = [s for s, _ in records]
        scores = [s for _, s in records]
        avg_score = np.mean(scores)

        # 🪵 Debug print (optional)
        print(f"🔍 {ticker} → Avg Sentiment Score: {avg_score:.3f}")

        if avg_score >= 0.98:  # ✅ Only keep perfect scores
            dominant = max(set(sentiments), key=sentiments.count)
            results.append({
                "Ticker": ticker,
                "Mentions": len(records),
                "Avg Sentiment Score": round(avg_score, 3),
                "Dominant Sentiment": dominant
            })

    # 🛡️ Handle empty result
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Mentions", ascending=False).reset_index(drop=True)
    return df


sentiment_df = score_sentiment_for_mentioned_tickers(news_df, known_companies, mentioned_tickers)
sentiment_df.to_csv("sentiment_summary.csv", index=False)
print(sentiment_df)

# 📈 Sort by Avg Sentiment Score (descending order)
sentiment_df = sentiment_df.sort_values("Avg Sentiment Score", ascending=False).reset_index(drop=True)

sentiment_df

import requests
import pandas as pd
from datetime import datetime, timedelta

# 🔑 Replace this with your actual API key
api_key = "79b7aa20b731467eb7965cced65acc54"

# 📆 Define the time range (last 30 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# 📰 Topic or stock to search for
query = "nifty50 OR stock market OR NSE India"

# 🌍 Base URL
url = "https://newsapi.org/v2/everything"

# 🧾 Parameters
params = {
    "q": query,
    "from": start_date.strftime('%Y-%m-%d'),
    "to": end_date.strftime('%Y-%m-%d'),
    "sortBy": "publishedAt",
    "language": "en",
    "pageSize": 100,
    "apiKey": api_key,
}

# 📡 Make the request
response = requests.get(url, params=params)
data = response.json()

# ✅ Convert to DataFrame
if data["status"] == "ok":
    articles = data["articles"]
    df = pd.DataFrame([{
        "title": article["title"],
        "publishedAt": article["publishedAt"],
        "description": article["description"],
        "url": article["url"],
        "source": article["source"]["name"]
    } for article in articles])

    print(f"✅ Retrieved {len(df)} news articles.")
    print(df.head())
    # Optional: Save to CSV
    df.to_csv("newsapi_last_30_days.csv", index=False)
else:
    print(f"❌ Error: {data.get('message')}")


# Fetch only the top 3 tickers with highest sentiment score from sentiment_df

# 📌 Sort by Avg Sentiment Score (descending) and pick top 3
top3_tickers_df = sentiment_df.sort_values("Avg Sentiment Score", ascending=False).head(3)

# 📌 Extract the Ticker list
tickers_to_fetch = top3_tickers_df["Ticker"].dropna().unique().tolist()

print(f"🎯 Top 3 Tickers Selected for Price Fetching: {tickers_to_fetch}")

tickers_to_fetch

import os
import pandas as pd
import yfinance as yf

# ✅ This function: Load CSV if exists, else fetch from Yahoo Finance
def load_or_fetch_historical_data(ticker, years=5):
    filename = f"historical_price_data_top3/{ticker}_price_history.csv"
    if os.path.exists(filename):
        print(f"📂 Loading existing historical data for {ticker}")
        df = pd.read_csv(filename)
    else:
        print(f"⬇️ Fetching new data for {ticker} (not found locally)")
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=years)
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
        df.reset_index(inplace=True)
        df = df[["Date", "Close"]].rename(columns={"Close": "Price"})
        os.makedirs("historical_price_data_top3", exist_ok=True)
        df.to_csv(filename, index=False)
    return df

# 📦 Load/Fecth Historical Data for Top 3
historical_data = {}

for ticker in tickers_to_fetch:
    historical_data[ticker] = load_or_fetch_historical_data(ticker)

# 👀 Display head(5) for each ticker
for ticker, df in historical_data.items():
    print(f"\n📈 {ticker} - First 5 Rows of Historical Data:\n")
    print(df.head(5))



# ✅ IMPORTS
import yfinance as yf
import pandas as pd
from nixtla import NixtlaClient
from utilsforecast.preprocessing import fill_gaps
import matplotlib.pyplot as plt
import os

# ✅ Setup TimeGPT Client
client = NixtlaClient(api_key="nixak-Cy1l2cVcBmGLFNxQGpF6g8XLJTWBUpVY3CIuZ4aKHaU2of7h7c6SRj0UD77hjR86HHdeYw06d05JIhbB")  # your key


# ✅ Imports already done above
# ✅ Assume TimeGPT client is already connected
# ✅ Assume tickers_to_fetch is already defined (like ['COP', 'IRM', 'NOC'])

# ✅ Updated and Correct Forecasting Function
def forecast_from_existing_csv(ticker, horizon=5):
    print(f"\n🔍 Processing {ticker}...")

    path = f"historical_price_data_top3/{ticker}_price_history.csv"
    if not os.path.exists(path):
        print(f"❌ File not found for {ticker}. Skipping.")
        return None

    # 📂 Step 1: Load CSV
    df = pd.read_csv(path)

    # 🧹 Step 2: Drop the first junk row
    df = df.drop(index=0).reset_index(drop=True)

    # 🧹 Step 3: Rename columns
    df = df.rename(columns={"Date": "ds", "Price": "y"})

    # 🧹 Step 4: Force 'y' to numeric type
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # 🧹 Step 5: Drop any remaining NaN
    df = df.dropna(subset=["ds", "y"])

    # 🧹 Step 6: Datetime conversion
    df['ds'] = pd.to_datetime(df['ds'])

    # 🧹 Step 7: Add unique_id column for TimeGPT
    df['unique_id'] = ticker
    df = df[['unique_id', 'ds', 'y']]

    # 🧹 Step 8: Fill missing dates and interpolate
    df_filled = fill_gaps(df, freq='D')
    df_filled['y'] = df_filled['y'].interpolate(method='linear', limit_direction='both')

    # 🔮 Step 9: Forecast using TimeGPT
    forecast_df = client.forecast(
        df=df_filled,
        h=horizon,
        freq='D',
        model='timegpt-1'
    )

    # ✅ Return forecasted DataFrame
    return forecast_df

# ✅ Now run for all tickers
forecasts = {}

for ticker in tickers_to_fetch:
    forecast_df = forecast_from_existing_csv(ticker)
    forecasts[ticker] = forecast_df

    # ✅ Print the forecasted values
    if forecast_df is not None:
        print(f"\n📈 Forecasted values for {ticker}:\n")
        print(forecast_df)


import matplotlib.pyplot as plt
import pandas as pd
import os

# ✅ New plotting function
def plot_historical_and_forecast(ticker, historical_folder="historical_price_data_top3/", forecast_horizon=5):
    print(f"\n📊 Plotting {ticker}...")

    # Load Historical CSV
    historical_path = os.path.join(historical_folder, f"{ticker}_price_history.csv")
    if not os.path.exists(historical_path):
        print(f"❌ Historical data not found for {ticker}. Skipping.")
        return

    hist_df = pd.read_csv(historical_path)

    # Drop junk first row
    hist_df = hist_df.drop(index=0).reset_index(drop=True)

    # Rename columns
    hist_df = hist_df.rename(columns={"Date": "ds", "Price": "y"})

    # Force y to numeric
    hist_df['y'] = pd.to_numeric(hist_df['y'], errors='coerce')
    hist_df = hist_df.dropna(subset=["ds", "y"])

    # Convert ds to datetime
    hist_df['ds'] = pd.to_datetime(hist_df['ds'])

    # Get forecasted data from dictionary
    forecast_df = forecasts.get(ticker)
    if forecast_df is None:
        print(f"❌ No forecast available for {ticker}. Skipping plot.")
        return

    # Create plot
    plt.figure(figsize=(18,6))

    # Plot historical
    plt.plot(hist_df['ds'], hist_df['y'], label='Historical Data', color='blue')

    # Plot forecasted (join from end of historical)
    plt.plot(forecast_df['ds'], forecast_df['TimeGPT'], label='Forecast', color='red', linestyle='--', marker='o')

    # Styling
    plt.title(f'{ticker} Stock Price: Historical + TimeGPT Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# ✅ Now plot for all tickers
for ticker in tickers_to_fetch:
    plot_historical_and_forecast(ticker)


