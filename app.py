from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Union
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

from tavily import TavilyClient
import yfinance as yf
import requests
import os
from datetime import date, timedelta, datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import json
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import pandas as pd
import ta
import dateparser
import pytz
import warnings

warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "agent"
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_........."
os.environ["TAVILY_API_KEY"] = "tvly-dev-.........."
os.environ["GROQ_API_KEY"] = "gsk.........."
os.environ["OPENAI_API_KEY"] = "sk-proj.........."

if not os.getenv("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY is not set.")
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set. Groq LLM will not work.")
    exit("Please set GROQ_API_KEY in your .env file.")
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY is not set. LangMem embeddings will not work.")
    exit("Please set OPENAI_API_KEY in your .env file.")
if not os.getenv("LANGSMITH_API_KEY"):
    print("WARNING: LANGSMITH_API_KEY is not set. LangSmith tracing will not work.")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]

@tool
def ratios(ticker: str, metric: str) -> str:
    try:
        if not ticker or not metric:
            return "Missing ticker or metric. Example input: {\"ticker\": \"AAPL\", \"metric\": \"trailingPE\"}"
        stock = yf.Ticker(ticker)
        info = stock.info
        value = info.get(metric)
        if value is not None:
            return f"The {metric} for {ticker.upper()} is {value}"
        else:
            return f"Could not find '{metric}' for {ticker.upper()}."
    except Exception as e:
        return f"Error fetching data: {e}"

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool
def tavily_search_tool(query: str) -> str:
    try:
        results = tavily.search(query=query, max_results=3)
        snippets = [res['content'] for res in results['results']]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Tavily error: {e}"

@tool
def compare_companies(tickers: List[str], metric: List[str]) -> str:
    if not tickers or not metric:
        return "Missing tickers or metrics. Example: {\"tickers\": [\"AAPL\", \"TSLA\"], \"metric\": [\"trailingPE\", \"marketCap\"]}"
    output = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            output.append(f"\n📊 **{ticker.upper()}**")
            for m in metric:
                value = info.get(m)
                if value is not None:
                    output.append(f" - {m}: {value}")
                else:
                    output.append(f" - {m}: Not available")
        except Exception as e:
            output.append(f"\n{ticker.upper()}: Error - {e}")
    return "\n".join(output)

def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': encoded_message}

def send_message(service, user_id, message):
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        return f"Email sent! Message ID: {sent_message['id']}"
    except Exception as e:
        return f"Error sending email: {e}"

@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    try:
        creds = Credentials.from_authorized_user_file("/content/token.json", scopes=['https://www.googleapis.com/auth/gmail.send'])
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        service = build('gmail', 'v1', credentials=creds)
        message = create_message("rajualbum295@gmail.com", to, subject, body)
        result = send_message(service, "me", message)
        return result
    except Exception as e:
        return f"Gmail tool error: {e}"

@tool
def create_calendar_event_tool(summary: str, start_time: str, end_time: str = None, description: str = "") -> str:
    try:
        creds = Credentials.from_authorized_user_file("/content/token.json", scopes=['https://www.googleapis.com/auth/calendar.events'])
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        service = build('calendar', 'v3', credentials=creds)
        summary = summary or "Untitled Event"
        description = description or ""
        start_dt = dateparser.parse(start_time, settings={'TIMEZONE': 'Asia/Kolkata', 'RETURN_AS_TIMEZONE_AWARE': True})
        if not start_dt and "next monday" in start_time.lower():
            today = datetime.now().astimezone()
            days_ahead = (0 - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            start_dt = today + timedelta(days=days_ahead)
            start_dt = start_dt.replace(hour=9, minute=0)
        if not start_dt:
            raise ValueError(f"Could not parse start_time: {start_time}")
        if end_time:
            end_dt = dateparser.parse(end_time, settings={'TIMEZONE': 'Asia/Kolkata', 'RETURN_AS_TIMEZONE_AWARE': True})
            if not end_dt:
                end_dt = start_dt + timedelta(hours=1)
        else:
            end_dt = start_dt + timedelta(hours=1)
        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()
        event = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_str},
            'end': {'dateTime': end_str}
        }
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Event created: {created_event.get('htmlLink')}"
    except Exception as e:
        return f"Calendar tool error: {e}"

@tool
def fetch_filtered_financial_news(symbol: str, days: int = 7):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    finnhub_api_key = "cvps0jpr01qve7iqspigcvps0jpr01qve7iqspj0"
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return pd.DataFrame()
    data = response.json()
    if not data:
        return pd.DataFrame()
    articles = [(item.get('headline', ''), item.get('url', '')) for item in data if 'headline' in item]
    filtered_articles = [(h, url) for h, url in articles if h and symbol.lower() in h.lower() and '?' not in h]
    if not filtered_articles:
        return pd.DataFrame()
    news_df = pd.DataFrame(filtered_articles, columns=["Headline", "URL"])
    return news_df["Headline"]

def basic(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1mo")
    current_price = info['currentPrice']
    one_day_change = ((hist['Close'][-1] - hist['Close'][-2]) / hist['Close'][-2]) * 100
    one_month_change = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]) * 100
    start_of_year = date(date.today().year, 1, 1)
    ytd_hist = stock.history(start=start_of_year)
    ytd_change = ((ytd_hist['Close'][-1] - ytd_hist['Close'][0]) / ytd_hist['Close'][0]) * 100
    return {
        "Current Price": f"${current_price:.2f}",
        "1-Day Change": f"{one_day_change:.2f}%",
        "1-Month Change": f"{one_month_change:.2f}%",
        "YTD Performance": f"{ytd_change:.2f}%",
        "52-Week Range": f"${info['fiftyTwoWeekLow']:.2f} – ${info['fiftyTwoWeekHigh']:.2f}",
        "Market Cap": f"${info['marketCap'] / 1e9:.0f}B"
    }

def get_technical_indicators(ticker):
    df = yf.Ticker(ticker).history(period="6mo").dropna()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    rsi = df['RSI'].iloc[-1]
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    ma50 = df['MA50'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    macd_calc = ta.trend.MACD(df['Close'])
    macd_line = macd_calc.macd().iloc[-1]
    signal_line = macd_calc.macd_signal().iloc[-1]
    macd_trend = "Bullish" if macd_line > signal_line else "Bearish"
    support = df['Close'].rolling(window=20).min().iloc[-1]
    resistance = df['Close'].rolling(window=20).max().iloc[-1]
    return {
        "RSI": f"{rsi:.2f} → {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}\n",
        "50-day MA": f"${ma50:.2f}",
        "200-day MA": f"${ma200:.2f}",
        "MACD": f"{macd_trend} trend",
        "Support Zone": f"${support*0.98:.2f} – ${support*1.02:.2f}",
        "Resistance Zone": f"${resistance*0.98:.2f} – ${resistance*1.02:.2f}"
    }

def earnings(ticker):
    ticker_obj = yf.Ticker(ticker)
    earnings_data = ticker_obj.earnings_dates
    return earnings_data

@tool
def analyze(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        basic_data = basic(ticker)
        tech_data = get_technical_indicators(ticker)
        earnings_df = earnings(ticker)
        upcoming_date = stock.calendar["Earnings Date"][0]
        report = f"""📊 ── BASIC COMPANY INFO ({ticker.upper()}) ──
• Current Price       : {basic_data['Current Price']}
• 1-Day Change        : {basic_data['1-Day Change']}
• 1-Month Change      : {basic_data['1-Month Change']}
• YTD Performance     : {basic_data['YTD Performance']}
• 52-Week Range       : {basic_data['52-Week Range']}
• Market Cap          : {basic_data['Market Cap']}

📈 ── FINANCIAL RATIOS ──
PE Ratio             : {info.get('trailingPE', 'N/A')}
PB Ratio             : {info.get('priceToBook', 'N/A')}
EPS                  : {info.get('trailingEps', 'N/A')}
ROE                  : {info.get('returnOnEquity', 'N/A')}
Market Cap           : {info.get('marketCap', 'N/A')}
Beta                 : {info.get('beta', 'N/A')}
Revenue              : {info.get('totalRevenue', 'N/A')}

📈 ── TECHNICAL INDICATORS ──
• RSI                 : {tech_data["RSI"]}
• 50-day MA           : {tech_data["50-day MA"]}
• 200-day MA          : {tech_data["200-day MA"]}
• MACD                : {tech_data["MACD"]}
• Support Zone        : {tech_data["Support Zone"]}
• Resistance Zone     : {tech_data["Resistance Zone"]}

🧾 ── EARNINGS ──
• Upcoming Earnings   : {upcoming_date.strftime('%Y-%m-%d')}
• Last 3 Earnings:
{earnings_df.to_markdown()}
"""
        return report
    except Exception as e:
        return f"❌ Error analyzing {ticker}: {str(e)}"
