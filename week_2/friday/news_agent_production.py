#!/usr/bin/env python3
"""
Week 2 - Friday: Production-Grade News Research Agent
With logging, retries, structured output, and error recovery.
"""

import os
import sys
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ============================================
# 1. LOGGING SETUP
# ============================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# 2. LOAD ENVIRONMENT
# ============================================
def load_env_file():
    possible_paths = [
        Path("/home/sadiaraja25/calderr-ai-2026/week_2/.env"),
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded .env from: {env_path}")
            return True
    logger.error("Could not find .env file")
    return False

load_env_file()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found!")
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

logger.info(f"GROQ_API_KEY loaded: {GROQ_API_KEY[:10]}...")
logger.info(f"NEWS_API_KEY: {'✅' if NEWS_API_KEY else '❌'}")
logger.info(f"WEATHER_API_KEY: {'✅' if WEATHER_API_KEY else '❌'}")

# ============================================
# 3. STRUCTURED OUTPUT WITH PYDANTIC
# ============================================

class NewsAnalysisReport(BaseModel):
    """Structured report from news analysis."""
    
    topic: str = Field(..., description="Main topic of the news")
    summary: str = Field(..., description="Brief summary of findings")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Entities extracted: people, organizations, locations"
    )
    confidence_score: float = Field(0.0, description="Confidence score (0-1)")

# ============================================
# 4. TOOLS WITH RETRY LOGIC
# ============================================

def call_with_retry(func, *args, max_retries: int = 3, delay: int = 2, **kwargs):
    """Execute a function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for {func.__name__}")
            result = func(*args, **kwargs)
            if "Error:" not in result:
                return result
            logger.warning(f"Attempt {attempt + 1} failed: {result[:50]}...")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} exception: {e}")
        
        if attempt < max_retries - 1:
            wait_time = delay * (2 ** attempt)
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return f"Error: All {max_retries} attempts failed"

@tool
def search_news(query: str, days: int = 7) -> str:
    """Search for news articles using NewsAPI with retry."""
    if not NEWS_API_KEY:
        return "Error: NEWS_API_KEY not set."
    
    def _search():
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query, "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 3, "apiKey": NEWS_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        if response.status_code != 200:
            return f"Error: API status {response.status_code}"
        
        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            return f"No news found for '{query}'."
        
        result = f"📰 News results for '{query}':\n\n"
        for i, article in enumerate(articles[:3], 1):
            title = article.get("title", "No title")[:100]
            source = article.get("source", {}).get("name", "Unknown")
            description = article.get("description", "No description")[:150]
            published = article.get("publishedAt", "").split("T")[0]
            result += f"{i}. {title}\n   Source: {source} | Date: {published}\n   {description}\n\n"
        return result
    
    return call_with_retry(_search)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city with retry."""
    if not WEATHER_API_KEY:
        return "Error: WEATHER_API_KEY not set."
    
    def _weather():
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHER_API_KEY, "q": city, "aqi": "no"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return f"Error: API status {response.status_code}"
        
        data = response.json()
        location = data.get("location", {})
        current = data.get("current", {})
        return f"🌤️ {location.get('name')}, {location.get('country')}:\n   Temp: {current.get('temp_c')}°C\n   Condition: {current.get('condition', {}).get('text')}"
    
    return call_with_retry(_weather)

@tool
def extract_entities(text: str) -> str:
    """Extract named entities from text using Groq."""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=GROQ_API_KEY)
        prompt = f"""Extract entities. Return ONLY valid JSON:
{{"people": [], "organizations": [], "locations": [], "dates": []}}
Text: {text[:300]}"""
        response = llm.invoke(prompt)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group())
            result = "🏷️ Entities:\n"
            result += f"   People: {', '.join(entities.get('people', ['None']))}\n"
            result += f"   Organizations: {', '.join(entities.get('organizations', ['None']))}\n"
            result += f"   Locations: {', '.join(entities.get('locations', ['None']))}"
            return result
        return "Could not extract entities."
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return f"Error: {str(e)}"

@tool
def classify_topic(text: str) -> str:
    """Classify the topic of a text."""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=GROQ_API_KEY)
        topics = ["Technology", "Business", "Politics", "Science", "Health", "Sports", "Entertainment", "Environment", "Education", "Other"]
        prompt = f"Classify into one category: {', '.join(topics)}\n\nText: {text[:300]}\n\nCategory:"
        response = llm.invoke(prompt)
        category = response.content.strip()
        return f"📌 Topic: {category if category in topics else 'Other'}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def summarize_text(text: str) -> str:
    """Summarize text concisely."""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=GROQ_API_KEY)
        prompt = f"Summarize in 2 sentences:\n{text[:400]}\n\nSummary:"
        response = llm.invoke(prompt)
        return f"📝 Summary: {response.content}"
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# 5. AGENT CLASS WITH LOGGING
# ============================================

class NewsResearchAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=GROQ_API_KEY)
        self.max_iterations = 3
        self.messages = []
        self.logger = logger
    
    def parse_tool_calls(self, response):
        tool_calls = []
        content = response.content if hasattr(response, 'content') else str(response)
        pattern = r'\[TOOL:\s*(\w+):\s*(.*?)\]'
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            tool_name = match[0].strip().lower()
            tool_input = match[1].strip()
            tool_calls.append({"name": tool_name, "args": {"input": tool_input}})
        self.logger.info(f"Parsed {len(tool_calls)} tool calls: {[tc['name'] for tc in tool_calls]}")
        return tool_calls
    
    def execute_tool(self, tool_name, tool_args):
        tool_input = tool_args.get("input", "")
        self.logger.info(f"Executing tool: {tool_name}({tool_input[:50]}...)")
        
        if tool_name == "search_news":
            return search_news.invoke(tool_input)
        elif tool_name == "get_weather":
            return get_weather.invoke(tool_input)
        elif tool_name == "extract_entities":
            return extract_entities.invoke(tool_input)
        elif tool_name == "classify_topic":
            return classify_topic.invoke(tool_input)
        elif tool_name == "summarize_text":
            return summarize_text.invoke(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    
    def run(self, user_query):
        self.logger.info(f"Starting query: {user_query[:100]}...")
        
        self.messages = [SystemMessage(content="""You are a news research agent. Tools:
1. search_news: Search news articles
2. get_weather: Get weather
3. extract_entities: Extract people, orgs, locations
4. classify_topic: Classify topic
5. summarize_text: Summarize content

Format: [TOOL: tool_name: input]""")]
        self.messages.append(HumanMessage(content=user_query))
        
        iteration = 0
        final_answer = None
        tool_results = []
        
        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"=== Iteration {iteration} ===")
            
            try:
                response = self.llm.invoke(self.messages)
                self.logger.info(f"LLM response length: {len(response.content)}")
            except Exception as e:
                self.logger.error(f"LLM call failed: {e}")
                return f"Error: LLM call failed - {e}", []
            
            tool_calls = self.parse_tool_calls(response)
            
            if not tool_calls:
                final_answer = response.content
                self.logger.info(f"Final answer found (length: {len(final_answer)})")
                break
            
            for tool_call in tool_calls[:2]:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                result = self.execute_tool(tool_name, tool_args)
                tool_results.append({"tool": tool_name, "result": result[:200]})
                
                self.messages.append(response)
                self.messages.append(ToolMessage(
                    content=result[:500],
                    tool_call_id=tool_call.get("id", "1")
                ))
        
        if not final_answer:
            final_answer = "Could not complete the research."
            self.logger.warning("No final answer found")
        
        self.logger.info(f"Query complete. {len(tool_results)} tools used.")
        return final_answer, tool_results

# ============================================
# 6. STREAMLIT UI WITH LOGGING
# ============================================

st.set_page_config(page_title="News Research Agent Pro", page_icon="📰", layout="wide")

st.title("📰 News Research Agent Pro")
st.markdown("Production-grade AI agent with logging, retries, and structured output")

with st.sidebar:
    st.header("⚙️ Settings")
    days = st.slider("Days to search", 1, 30, 7)
    st.header("🔑 API Status")
    st.success("✅ Groq API") if GROQ_API_KEY else st.error("❌ Groq API")
    st.success("✅ NewsAPI") if NEWS_API_KEY else st.error("❌ NewsAPI")
    st.success("✅ WeatherAPI") if WEATHER_API_KEY else st.error("❌ WeatherAPI")
    
    st.header("📋 Logs")
    if st.button("View Logs"):
        try:
            with open("logs/agent.log", "r") as f:
                st.code(f.read()[-1000:], language="log")
        except:
            st.info("No logs yet")

query = st.text_input("🔍 What news are you looking for?", placeholder="e.g., artificial intelligence, climate change...")

if st.button("🚀 Search News", type="primary", key="search_btn"):
    if query:
        with st.spinner("🔍 Researching news..."):
            try:
                agent = NewsResearchAgent()
                answer, tool_results = agent.run(query)
                
                st.markdown("### 📊 Results")
                for tr in tool_results:
                    with st.expander(f"🔧 {tr['tool'].replace('_', ' ').title()}"):
                        st.code(tr['result'], language="text")
                
                st.markdown("### 💬 Final Analysis")
                st.info(answer)
                st.caption(f"Tools used: {len(tool_results)}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")
    else:
        st.warning("Please enter a search query.")

st.markdown("---")
st.caption("Production Agent v1.0 | Logs: logs/agent.log")
