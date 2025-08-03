
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
import os
warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings

import time
import random
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "agent"
os.environ["LANGSMITH_API_KEY"] = "lsv2_.........."

os.environ["TAVILY_API_KEY"] = "tvly............"
os.environ["GROQ_API_KEY"] = "gsk_................."
os.environ["OPENAI_API_KEY"] = "sk-proj-..............."



if not os.getenv("LANGSMITH_API_KEY"):
    print("WARNING: LANGSMITH_API_KEY is not set. LangSmith tracing will not work.")

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

# === Define state ===
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]

# === Tool 1: Ratios (YFinance) ===
@tool
def ratios(ticker: str, metric: str) -> str:
    """
    Input: ticker (e.g., "AAPL") and metric (e.g., "trailingPE")
    Output: Value of metric from yfinance
    """
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


# === Tool 2: Tavily Search ===
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
@tool
def tavily_search_tool(query: str) -> str:
    """Search general knowledge questions using Tavily Web Search"""
    try:
        results = tavily.search(query=query, max_results=3)
        snippets = [res['content'] for res in results['results']]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Tavily error: {e}"


# === Tool 3: Comparing different companies ===
@tool
def compare_companies(tickers: List[str], metric: List[str]) -> str:
    """
    Compare multiple financial metrics across multiple tickers.

    Input:
        - tickers: List of stock tickers (e.g., ["AAPL", "TSLA"])
        - metric: List of YFinance keys (e.g., ["trailingPE", "marketCap", "totalRevenue"])

    Output:
        - Comparison of each metric for each ticker
    """
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


# === Tool 4: Send Email ===
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
    """
    Sends an email via Gmail API. Input: to, subject, body
    """
    try:
        print("📤 Loading credentials...")
        creds = Credentials.from_authorized_user_file("/content/token.json", scopes=['https://www.googleapis.com/auth/gmail.send'])

        if creds.expired and creds.refresh_token:
            print("🔄 Token expired, refreshing...")
            creds.refresh(Request())

        print("📬 Building Gmail service...")
        service = build('gmail', 'v1', credentials=creds)

        print("✉️ Creating message...")
        message = create_message("rajualbum295@gmail.com", to, subject, body)

        print("🚀 Sending message...")
        result = send_message(service, "me", message)

        print("✅ Done. Returning result.")
        return result

    except Exception as e:
        print("❌ Exception inside send_email_tool:", str(e))
        return f"Gmail tool error: {e}"

# === Tool 5: Create Calendar Event ===
@tool
def create_calendar_event_tool(summary: str, start_time: str, end_time: str = None, description: str = "") -> str:
    """
    Creates a Google Calendar event.
    Accepts natural language for start_time and end_time (e.g., 'tomorrow 2 PM').
    If end_time is not provided, defaults to 1 hour after start_time.
    """
    try:
        print("📅 Loading calendar credentials...")
        creds = Credentials.from_authorized_user_file(
            "/content/token.json",
            scopes=['https://www.googleapis.com/auth/calendar.events']
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        service = build('calendar', 'v3', credentials=creds)

        summary = summary or "Untitled Event"
        description = description or ""

  
        start_dt = dateparser.parse(start_time, settings={
            'TIMEZONE': 'Asia/Kolkata',
            'RETURN_AS_TIMEZONE_AWARE': True
        })

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
            end_dt = dateparser.parse(end_time, settings={
                'TIMEZONE': 'Asia/Kolkata',
                'RETURN_AS_TIMEZONE_AWARE': True
            })
            if not end_dt:
                print("⚠️ Could not parse end_time. Using default duration of 1 hour.")
                end_dt = start_dt + timedelta(hours=1)
        else:
            end_dt = start_dt + timedelta(hours=1)

        start_str = start_dt.isoformat()
        end_str = end_dt.isoformat()

        print(f"🕒 Creating event from {start_str} to {end_str}")
        print(f"📅 Event Summary: {summary}")
        print(f"📝 Description: {description}")

        event = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_str},
            'end': {'dateTime': end_str}
        }

        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ Event created: {created_event.get('htmlLink')}"

    except Exception as e:
        print("❌ Exception inside calendar tool:", str(e))
        return f"Calendar tool error: {e}"


# === Tool 6: Fetch Filtered Financial News ===
@tool
def fetch_filtered_financial_news(symbol: str, days: int = 7): 
    "fetch latest company news "
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    finnhub_api_key = "cvps0jpr01qve7iqspigcvps0jpr01qve7iqspj0" 
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text}")
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

# === Helper Functions for Analyze Tool ===
def basic(ticker):
    "analyze a specific company based on details"
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
    df = yf.Ticker(ticker).history(period="6mo")
    df = df.dropna()

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
    earnings_date = ticker_obj.calendar['Earnings Date'][0]  
    earnings_data = ticker_obj.earnings_dates 
    return earnings_data



@tool
def analyze(ticker: str) -> str:
    """
    Analyze a stock using three methods: basic financials, technical indicators, and earnings.
    Returns a detailed report combining all three insights.
    """
    try:
        basic_data = basic(ticker)
        tech_data = get_technical_indicators(ticker)
        earnings_df = earnings(ticker)

        upcoming_date = yf.Ticker(ticker).calendar["Earnings Date"][0]

        pe_ratio = ratios({"ticker": ticker, "metric": "trailingPE"})
        pb_ratio = ratios({"ticker": ticker, "metric": "priceToBook"})
        eps = ratios({"ticker": ticker, "metric": "trailingEps"})
        roe = ratios({"ticker": ticker, "metric": "returnOnEquity"})
        market_cap = ratios({"ticker": ticker, "metric": "marketCap"})
        dividend_yield = ratios({"ticker": ticker, "metric": "dividendYield"})
        beta = ratios({"ticker": ticker, "metric": "beta"})
        revenue = ratios({"ticker": ticker, "metric": "totalRevenue"})

        # ========== Build Full Report ==========
        report = f"""📊 ── BASIC COMPANY INFO ({ticker.upper()}) ──
• Current Price       : {basic_data['Current Price']}
• 1-Day Change        : {basic_data['1-Day Change']}
• 1-Month Change      : {basic_data['1-Month Change']}
• YTD Performance     : {basic_data['YTD Performance']}
• 52-Week Range       : {basic_data['52-Week Range']}
• Market Cap          : {basic_data['Market Cap']}

📈 ── Financial ratios ──
• PE Ratio (P/E)      : {pe_ratio}
• PB Ratio (P/B)      : {pb_ratio}
• EPS                 : {eps}
• ROE                 : {roe}
• Market Cap (Ratio)  : {market_cap}
• Dividend Yield      : {dividend_yield}
• Beta                : {beta}
• Revenue             : {revenue}

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

tools = [
    ratios,
    tavily_search_tool,
    compare_companies,
    send_email_tool,
    create_calendar_event_tool,
    fetch_filtered_financial_news,
    analyze]

llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY")).bind_tools(tools)


def model_node(state: AgentState) -> AgentState:
    response = safe_invoke_with_retry(llm, state["messages"])
    state["messages"].append(response)
    return state


def model_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(response)
    return state


def router(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tool"
    return END 

def tool_node_wrapper(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]



        print(f"tool we used: {tool_name}")
        print(f"input going into tool: {tool_input}")

        tool_func = next((t for t in tools if t.name == tool_name), None)

        if tool_func:
          result = tool_func.invoke(tool_func, tool_input)

        else:
            result = f"Error: Tool '{tool_name}' not found."

        state["messages"].append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
        )
    return state



# === LangGraph Workflow ===
workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("call_tool", tool_node_wrapper)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", router)
workflow.add_edge("call_tool", "model")
app = workflow.compile()

system_prompt = SystemMessage(content="""
You are a specialized Financial AI Assistant designed to assist users with financial analysis, stock research, news insights, and productivity tasks. Your core responsibility is to clearly understand the user's intent, select the most appropriate tool (or no tool if not needed), and deliver friendly, accurate, and contextually-aware responses.

Your behavior is governed by the following core principles:

---
🔍 INTENT CLASSIFICATION:

Before taking any action, classify the user request into one of these categories:

1. **Stock Metric Lookup** → use `ratios`
2. **Single Stock Deep Analysis** → use `analyze` (followed by summarization & optional email)
3. **Compare Stocks** → use `compare_companies`
4. **General Finance Question or News** → use `tavily_search_tool`
5. **Recent News on a Stock** → use `fetch_filtered_financial_news`
6. **Send Email** → use `send_email_tool`
7. **Create Calendar Event** → use `create_calendar_event_tool`
8. **Polite or Meta Requests** (e.g., greetings, "can you do X?") → No tool call

---
🛠 TOOL USAGE GUIDELINES:

1. **`ratios`**
   Use when the user asks about a financial metric of a specific company.
   Extract:
   - Ticker (e.g., Tesla → TSLA)
   - Metric → convert into YFinance format:
        - PE ratio → trailingPE
        - EPS → trailingEps
        - ROE → returnOnEquity
        - PB ratio → priceToBook
        - Market Cap → marketCap
        - Dividend Yield → dividendYield
        - Beta → beta
        - Revenue → totalRevenue

2. **`analyze`**
## When to use:
Use this tool when the user requests deep insights or a full analysis of a stock. Examples:
- "Analyze this stock"
- "Give full insights on Tesla"
- "Should I invest in Apple?"

## After the tool returns:
DO NOT paste the full raw output directly.
Instead, extract and summarize these key sections in a clear, engaging, and conversational **summary** of your analysis:
   - Current price
   - P/E and P/B ratios
   - ROE and EPS
   - Market cap or volatility/beta
   - Key technical indicators (RSI, MACD, MAs)
   - Upcoming earnings or past earnings surprises
   Summarize what this means for both short-term and long-term investors

End with:
   > “Would you like me to send the full detailed report to your email?”

## If the user agrees:
Call the `send_email_tool` with:
- `subject`: "<Company Name> Stock Analysis"
- `to`: the provided email or ts4044903@gmail.com
- In **body** pass the full summary into the body field.


3. **`compare_companies`**
   Use when user asks to compare two or more companies (with or without metrics).
   Compare these 5 metrics by default:
   - PE ratio (`trailingPE`)
   - Market Cap (`marketCap`)
   - Revenue (`totalRevenue`)
   - ROE (`returnOnEquity`)
   - Dividend Yield (`dividendYield`)

4. **`tavily_search_tool`**
   Use for:
   - General finance questions
   - Definitions (e.g., "What is ROE?")
   - Market or economy-related queries
   - News-style queries (when company is mentioned without asking for financials)

5. **`fetch_filtered_financial_news`**
   Use when user explicitly wants “latest news” for a company.
   - Default days = 7 if user doesn’t specify.
   - Return recent filtered headlines only.

6. **`send_email_tool`**
   Only trigger if user:
   - Provides email ID
   - Says "send email"
   - Responds yes to your email offer after analysis summary
   Use default `to`: ts4044903@gmail.com if none provided.

7. `create_calendar_event_tool` — for scheduling calendar events. Format:
    {{
        "summary": "Meeting with investor",
        "start_time": "2025-07-22T10:00:00+05:30",
        "end_time": "2025-07-22T11:00:00+05:30",
        "description": "Discuss Q2 results"
    }}

 Notes:
Only start_time is required. summary, end_time, and description are optional.

Use from datetime import datetime, timedelta to resolve relative or natural expressions, such as:
“next Monday”
“after 3 hours”
“day after tomorrow”
You can also use timezone-aware datetime strings (e.g., +05:30 for IST).

🧠 Smart Handling Instructions:
If the user only mentions a date/day and not the time, politely confirm:
"Sure! What time would you like me to schedule it?"

If the user says something like:
“Schedule a meeting next Monday” → Use datetime.today() + logic to compute the next Monday.
“Book a meeting 2 days later at 3 PM” → Add timedelta(days=2) to today's date.
If the duration (i.e., end time) is not provided, assume a default of 1 hour unless user specifies otherwise.

---
🧠 MEMORY & RETRIEVAL:

You have access to a conversation history retriever. Use retrieved content as private context only. Never repeat or read the retrieved memory aloud.

Use retrieval to:
- Understand prior conversation
- Maintain continuity
- Resolve pronouns and incomplete references (e.g., “compare that one too”, “analyze again”)

---
🚫 WHEN **NOT** TO CALL TOOLS:

- If the user is just greeting, thanking, or saying goodbye
- If the user is asking if you're capable (e.g., “can you compare stocks?”), just confirm politely

✅ Sample responses:
- “Yes, I can help compare companies. Which ones would you like me to compare?”
- “Sure! I can help with that. Please give me the stock symbol or company name.”

---
🗣️ STYLE & TONE:

- Keep it conversational, friendly, and informative — not robotic.
- Prefer concise explanations over large data dumps.
- For long output like `analyze`, summarize smartly first, then offer to email the full report.

---

If the user request involves multiple metrics or multiple tool calls (e.g., both PE ratio and ROE),
 wait for all relevant tool results to return, then summarize them together in one reply.


""")


def log_to_file(user_input: str, ai_response: str, file_path: str = "conversation_log.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\nUser: {user_input}\n")
        f.write(f"AI  : {ai_response}\n")
        f.write("-" * 50 + "\n")

def interactive_conversation():
    messages = [system_prompt]
    print("📈 Financial AI Agent started. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        if not user_input:
            continue

        last_user_ai_turn = []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_user_ai_turn.insert(0, msg)
            elif isinstance(msg, HumanMessage):
                last_user_ai_turn.insert(0, msg)
                break

        current_turn_messages = [system_prompt] + last_user_ai_turn + [HumanMessage(content=user_input)]
        state = {"messages": current_turn_messages}


        final_state = app.invoke(state)
        messages = final_state["messages"]

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"AI: {msg.content}\n")
                break

if __name__ == "__main__":
    interactive_conversation()
