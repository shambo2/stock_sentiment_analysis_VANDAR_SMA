import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from pandas.tseries.offsets import BDay

# Project overview
st.write("""
    # Overview:book:
##### Stock Trading Strategy Using Sentiment Analysis and Simple Moving Averages (SMA)

In this project, I’ve combined sentiment analysis and technical analysis to create a stock trading strategy, demonstrating how modern NLP techniques and traditional stock market indicators can work together to generate buy and sell signals.

I used VADER sentiment analysis to gauge the tone of news headlines (positive, neutral, or negative). This helps predict market movements, with positive sentiment suggesting a buy and negative sentiment signaling a sell.

For technical analysis, I implemented Simple Moving Averages (SMA) using 2-day and 5-day SMAs. A crossover of the short-term SMA (2-day) above the medium-term SMA (5-day) indicates a buy, while a downward crossover suggests a sell.
""")
# Sidebar with dropdown options
st.sidebar.header("Options")
option = st.sidebar.selectbox(
    "Choose an option",
    ("Overview", "SMA Trade Calls", "VADER Trade Calls", "Merged Trade Calls")
)

# Load data
data_amd = yf.download('AMD', '2024-08-28', end='2024-09-27')
data_amd['2_SMA'] = data_amd['Close'].rolling(window=2).mean()
data_amd['5_SMA'] = data_amd['Close'].rolling(window=5).mean()
data_amd = data_amd[data_amd['5_SMA'].notna()]

# SMA trade calls
Trade_Buy = []
Trade_Sell = []
for i in range(len(data_amd)-1):
    if ((data_amd['2_SMA'].values[i] < data_amd['5_SMA'].values[i]) & (data_amd['2_SMA'].values[i+1] > data_amd['5_SMA'].values[i+1])):
        Trade_Buy.append(i)
    elif ((data_amd['2_SMA'].values[i] > data_amd['5_SMA'].values[i]) & (data_amd['2_SMA'].values[i+1] < data_amd['5_SMA'].values[i+1])):
        Trade_Sell.append(i)

# Create Plotly figure for SMA trade calls
fig_sma = go.Figure()

# Add Close price trace
fig_sma.add_trace(go.Scatter(x=data_amd.index, y=data_amd['Close'], mode='lines', name='Close'))

# Add 2_SMA trace
fig_sma.add_trace(go.Scatter(x=data_amd.index, y=data_amd['2_SMA'], mode='lines', name='2_SMA'))

# Add 5_SMA trace
fig_sma.add_trace(go.Scatter(x=data_amd.index, y=data_amd['5_SMA'], mode='lines', name='5_SMA'))

# Add Buy signals
fig_sma.add_trace(go.Scatter(x=data_amd.index[Trade_Buy], y=data_amd['Close'].iloc[Trade_Buy], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))

# Add Sell signals
fig_sma.add_trace(go.Scatter(x=data_amd.index[Trade_Sell], y=data_amd['Close'].iloc[Trade_Sell], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))

# Update layout
fig_sma.update_layout(
    title='Trade Calls - Moving Averages Crossover',
    xaxis_title='Date',
    yaxis_title='Price in Dollars',
    xaxis=dict(tickangle=60),
    template='plotly_white'
)

# Sentiment analysis
url = 'https://newsapi.org/v2/everything?'
parameters = {
    'q': 'AMD',
    'sortBy': 'popularity',
    'pageSize': 100,
    'apiKey': 'b5609d2514664fdd8345d0c8f83267cb',
}
response = requests.get(url, params=parameters)
data = pd.DataFrame(response.json())
news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1)
final_news = news_df.loc[:, ['publishedAt', 'title']]
final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
final_news.sort_values(by='publishedAt', inplace=True)

def get_trade_open(date):
    curr_date_open = pd.to_datetime(date).floor('d').replace(hour=13, minute=30) - BDay(0)
    curr_date_close = pd.to_datetime(date).floor('d').replace(hour=20, minute=0) - BDay(0)
    prev_date_close = (curr_date_open - BDay()).replace(hour=20, minute=0)
    next_date_open = (curr_date_close + BDay()).replace(hour=13, minute=30)
    if ((pd.to_datetime(date) >= prev_date_close) & (pd.to_datetime(date) < curr_date_open)):
        return curr_date_open
    elif ((pd.to_datetime(date) >= curr_date_close) & (pd.to_datetime(date) < next_date_open)):
        return next_date_open
    else:
        return None

final_news["trading_time"] = final_news["publishedAt"].apply(get_trade_open)
final_news = final_news[pd.notnull(final_news['trading_time'])]
final_news['Date'] = pd.to_datetime(pd.to_datetime(final_news['trading_time']).dt.date)

analyzer = SentimentIntensityAnalyzer()
cs = []
for row in range(len(final_news)):
    cs.append(analyzer.polarity_scores(final_news['title'].iloc[row])['compound'])
final_news['compound_vader_score'] = cs
final_news = final_news[(final_news[['compound_vader_score']] != 0).all(axis=1)].reset_index(drop=True)

unique_dates = final_news['Date'].unique()
grouped_dates = final_news.groupby(['Date'])
keys_dates = list(grouped_dates.groups.keys())

max_cs = []
min_cs = []
for key in grouped_dates.groups.keys():
    data = grouped_dates.get_group(key)
    if data["compound_vader_score"].max() > 0:
        max_cs.append(data["compound_vader_score"].max())
    elif data["compound_vader_score"].max() < 0:
        max_cs.append(0)
    if data["compound_vader_score"].min() < 0:
        min_cs.append(data["compound_vader_score"].min())
    elif data["compound_vader_score"].min() > 0:
        min_cs.append(0)

extreme_scores_dict = {'Date': keys_dates, 'max_scores': max_cs, 'min_scores': min_cs}
extreme_scores_df = pd.DataFrame(extreme_scores_dict)

final_scores = []
for i in range(len(extreme_scores_df)):
    final_scores.append(extreme_scores_df['max_scores'].values[i] + extreme_scores_df['min_scores'].values[i])
extreme_scores_df['final_scores'] = final_scores

vader_Buy = []
vader_Sell = []
for i in range(len(extreme_scores_df)):
    if extreme_scores_df['final_scores'].values[i] > 0.20:
        vader_Buy.append(extreme_scores_df['Date'].iloc[i].date())
    elif extreme_scores_df['final_scores'].values[i] < -0.20:
        vader_Sell.append(extreme_scores_df['Date'].iloc[i].date())

vader_buy = []
for i in range(len(data_amd)):
    if data_amd.index[i].date() in vader_Buy:
        vader_buy.append(i)

vader_sell = []
for i in range(len(data_amd)):
    if data_amd.index[i].date() in vader_Sell:
        vader_sell.append(i)

# Create Plotly figure for VADER trade calls
fig_vader = go.Figure()

# Add Close price trace
fig_vader.add_trace(go.Scatter(x=data_amd.index, y=data_amd['Close'], mode='lines', name='Close'))

# Add Buy signals
fig_vader.add_trace(go.Scatter(x=data_amd.index[vader_buy], y=data_amd['Close'].iloc[vader_buy], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))

# Add Sell signals
fig_vader.add_trace(go.Scatter(x=data_amd.index[vader_sell], y=data_amd['Close'].iloc[vader_sell], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))

# Update layout
fig_vader.update_layout(
    title='Trade Calls - VADER',
    xaxis_title='Date',
    yaxis_title='Price in Dollars',
    xaxis=dict(tickangle=60),
    template='plotly_white'
)

# Prioritizing SMA signals
final_buy = list(set(Trade_Buy + vader_buy) - set(Trade_Sell))
final_sell = list(set(Trade_Sell + vader_sell) - set(Trade_Buy))

# Create Plotly figure for merged trade calls
fig_merged = go.Figure()

# Add Close price trace
fig_merged.add_trace(go.Scatter(x=data_amd.index, y=data_amd['Close'], mode='lines', name='Close'))

# Add 2_SMA trace
fig_merged.add_trace(go.Scatter(x=data_amd.index, y=data_amd['2_SMA'], mode='lines', name='2_SMA'))

# Add 5_SMA trace
fig_merged.add_trace(go.Scatter(x=data_amd.index, y=data_amd['5_SMA'], mode='lines', name='5_SMA'))

# Add Buy signals
fig_merged.add_trace(go.Scatter(x=data_amd.index[final_buy], y=data_amd['Close'].iloc[final_buy], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))

# Add Sell signals
fig_merged.add_trace(go.Scatter(x=data_amd.index[final_sell], y=data_amd['Close'].iloc[final_sell], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))

# Update layout
fig_merged.update_layout(
    title='Trade Calls - MERGED',
    xaxis_title='Date',
    yaxis_title='Price in Dollars',
    xaxis=dict(tickangle=60),
    template='plotly_white'
)

# Display plots with descriptions
st.write("## SMA Trade Calls")
st.write("This section shows the trade calls based on Simple Moving Averages (SMA).")
st.plotly_chart(fig_sma)

st.write("## VADER Trade Calls")
# Display headlines and VADER scores
st.plotly_chart(fig_vader)

# Create a checkbox to toggle the display of DataFrames
show_dataframes = st.checkbox('Show Data of Articles & VADER Scores')

if show_dataframes:
    # Display the final_news DataFrame
    st.write(final_news[['publishedAt', 'title', 'compound_vader_score']])

    # Display the extreme_scores_df DataFrame
    st.write("##### Extreme VADER Scores by Date")
    st.write(extreme_scores_df)
st.write("""
To gather news headlines for the project, I used NewsAPI, which provides access to a wide range of news sources. I requested headlines related to specific stocks (In this case AMD) over a set time frame, pulling the title, description, and publication date.

To calculate the sentiment score, I used VADER (Valence Aware Dictionary and Sentiment Reasoner), a pre-trained model designed for text sentiment analysis. VADER assigns a compound score to each headline, ranging from -1 (most negative) to +1 (most positive). This score allows us to quantify the sentiment of each news article, which we then use to analyze its potential impact on stock price movements. 

In addition to calculating the sentiment score, I identified extreme VADER scores by focusing on particularly strong sentiment signals. A score above 0.20 is considered highly positive and treated as a potential buy signal, while a score below -0.20 is considered highly negative, signaling a potential sell. These extreme scores allow us to filter out neutral or mild sentiment and focus on market-moving news events.
""")

st.write("## Merged Trade Calls")
st.write("This section shows the merged trade calls from both SMA and VADER analysis.")
st.plotly_chart(fig_merged)

# Add an animated arrow to indicate more content
st.markdown(
    """
    <div style="position: fixed; bottom: 20px; width: 100%; text-align: center;">
        <span id="scroll-arrow" style="font-size: 48px; color: gray; animation: fadeInOut 2s infinite;">
            ↓
        </span>
    </div>
    <style>
    @keyframes fadeInOut {
        0%, 100% {
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
    }
    </style>
    <script>
    window.addEventListener('scroll', function() {
        var arrow = document.getElementById('scroll-arrow');
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight) {
            arrow.style.display = 'none';
        } else {
            arrow.style.display = 'block';
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)