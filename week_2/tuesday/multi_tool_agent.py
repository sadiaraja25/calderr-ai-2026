import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# ============================================
# LOAD ENVIRONMENT
# ============================================
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found!")
    exit(1)

# ============================================
# STEP 1: DEFINE 5+ TOOLS
# ============================================

@tool
def web_search_mock(query: str) -> str:
    """
    Mock web search tool.
    
    Args:
        query: The search query
    """
    knowledge_base = {
        "rag": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with LLMs.",
        "llm": "LLM stands for Large Language Model. It's a neural network trained on text data.",
        "agent": "An AI agent is a system that uses LLMs to perform tasks with tools and memory.",
        "python": "Python is a high-level programming language created by Guido van Rossum.",
        "transformer": "Transformer is a neural network architecture using attention mechanisms."
    }
    
    for key, value in knowledge_base.items():
        if key in query.lower():
            return f"Search result for '{query}': {value}"
    
    return f"Search result for '{query}': No specific information found."

@tool
def calculate(expression: str) -> str:
    """
    Calculate math expressions.
    
    Args:
        expression: Math expression (e.g., "2 + 2")
    """
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def summarize_text(text: str) -> str:
    """
    Summarize text using LLM.
    
    Args:
        text: Text to summarize
    """
    # This would normally call an LLM, but we'll mock it
    words = text.split()
    if len(words) > 20:
        return f"Summary of {len(words)} words: {text[:100]}..."
    return f"Text summary: {text}"

@tool
def classify_sentiment(text: str) -> str:
    """
    Classify the sentiment of text.
    
    Args:
        text: Text to analyze
    """
    # Mock sentiment analysis
    positive_words = ["good", "great", "amazing", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "worst", "poor"]
    
    text_lower = text.lower()
    if any(word in text_lower for word in positive_words):
        return "Sentiment: Positive"
    elif any(word in text_lower for word in negative_words):
        return "Sentiment: Negative"
    return "Sentiment: Neutral"

@tool
def get_weather(city: str) -> str:
    """
    Get weather for a city.
    
    Args:
        city: City name
    """
    weather_data = {
        "london": "15°C, sunny",
        "paris": "18°C, cloudy",
        "new york": "22°C, clear",
        "tokyo": "25°C, humid",
        "sydney": "20°C, partly cloudy"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

# ============================================
# STEP 2: TOOL REGISTRY
# ============================================

TOOLS = {
    "search": web_search_mock,
    "calculate": calculate,
    "date": get_current_date,
    "summarize": summarize_text,
    "sentiment": classify_sentiment,
    "weather": get_weather
}

TOOL_DESCRIPTIONS = """
Available tools:
1. search: Search knowledge base. Input: query
2. calculate: Do math. Input: expression (e.g., "2 + 2")
3. date: Get current date. No input needed
4. summarize: Summarize text. Input: text
5. sentiment: Classify sentiment. Input: text
6. weather: Get weather for a city. Input: city name

Format:
To use a tool: [TOOL: tool_name: tool_input]
To give final answer: [FINAL: your answer]
"""

# ============================================
# STEP 3: AGENT WITH TOOL ROUTING
# ============================================

class MultiToolAgent:
    def __init__(self):
        self.max_iterations = 5
        self.messages = [SystemMessage(content=TOOL_DESCRIPTIONS)]
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
    
    def parse_output(self, output):
        """Parse LLM output for tool calls."""
        # Check for final answer
        if "[FINAL:" in output:
            final_match = re.search(r'\[FINAL:\s*(.*?)\]', output, re.DOTALL)
            if final_match:
                return {"type": "final", "content": final_match.group(1).strip()}
        
        # Check for tool call
        if "[TOOL:" in output:
            tool_match = re.search(r'\[TOOL:\s*(\w+):\s*(.*?)\]', output, re.DOTALL)
            if tool_match:
                return {
                    "type": "tool",
                    "tool_name": tool_match.group(1).strip(),
                    "tool_input": tool_match.group(2).strip()
                }
        
        return {"type": "message", "content": output}
    
    def execute_tool(self, tool_name, tool_input):
        """Execute the requested tool."""
        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name].invoke(tool_input)
                return f"Result: {result}"
            except Exception as e:
                return f"Tool error: {e}"
        return f"Unknown tool: {tool_name}"
    
    def run(self, user_query):
        """Run the agent loop."""
        print(f"\n🧑 User: {user_query}")
        print("-" * 50)
        
        self.messages.append(HumanMessage(content=user_query))
        iteration = 0
        final_answer = None
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            response = self.llm.invoke(self.messages)
            output = response.content
            print(f"💭 LLM: {output[:200]}...")
            
            parsed = self.parse_output(output)
            
            if parsed["type"] == "final":
                final_answer = parsed["content"]
                print(f"✅ Final: {final_answer}")
                break
            
            elif parsed["type"] == "tool":
                tool_name = parsed["tool_name"]
                tool_input = parsed["tool_input"]
                print(f"🔧 Tool Call: {tool_name}({tool_input})")
                
                result = self.execute_tool(tool_name, tool_input)
                print(f"📊 Result: {result}")
                
                self.messages.append(response)
                self.messages.append(HumanMessage(content=f"Tool result: {result}"))
            
            else:
                self.messages.append(response)
        
        if not final_answer:
            final_answer = "Could not complete the task."
        
        return final_answer

# ============================================
# STEP 4: TEST THE AGENT
# ============================================

def run_tests():
    print("=" * 60)
    print("🤖 LAB 2.2: MULTI-TOOL RESEARCH AGENT")
    print("=" * 60)
    
    agent = MultiToolAgent()
    
    test_queries = [
        "What is RAG?",
        "Calculate 25 * 4",
        "What's the current date?",
        "Classify sentiment: I love this product!",
        "What's the weather in London?"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        result = agent.run(query)
        print(f"\n📝 Final Answer: {result}")
        print("=" * 60)

if __name__ == "__main__":
    run_tests()
