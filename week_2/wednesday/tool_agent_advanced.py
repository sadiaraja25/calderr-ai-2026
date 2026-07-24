#!/usr/bin/env python3
"""
Week 2 - Wednesday: Advanced Tool Calling Agent
Build a 5-tool agent: search_db, calculate, format_date, convert_currency, summarize_text.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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

print(f"✅ GROQ_API_KEY loaded: {GROQ_API_KEY[:10]}...")

# ============================================
# STEP 1: DEFINE 5 TOOLS
# ============================================

# Tool 1: Search Database
@tool
def search_db(query: str) -> str:
    """
    Search the knowledge database for information.
    
    Args:
        query: The search query
    """
    knowledge_base = {
        "rag": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with LLMs.",
        "llm": "LLM stands for Large Language Model. It's a neural network trained on text data.",
        "agent": "An AI agent is a system that uses LLMs to perform tasks with tools and memory.",
        "transformer": "Transformer is a neural network architecture using attention mechanisms.",
        "embedding": "Embeddings are numerical representations of text that capture meaning."
    }
    
    for key, value in knowledge_base.items():
        if key in query.lower():
            return f"Search result: {value}"
    
    return f"No information found for '{query}'"

# Tool 2: Calculator
@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: Math expression (e.g., "2 + 2", "25 * 4")
    """
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"

# Tool 3: Format Date
@tool
def format_date(format_type: str) -> str:
    """
    Get the current date in a specific format.
    
    Args:
        format_type: The format type ('short', 'long', 'iso', 'us')
    """
    now = datetime.now()
    
    formats = {
        "short": now.strftime("%Y-%m-%d"),
        "long": now.strftime("%B %d, %Y"),
        "iso": now.isoformat(),
        "us": now.strftime("%m/%d/%Y"),
        "european": now.strftime("%d/%m/%Y")
    }
    
    return formats.get(format_type.lower(), formats["short"])

# Tool 4: Convert Currency
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert currency using mock exchange rates.
    
    Args:
        amount: The amount to convert
        from_currency: Source currency code (USD, EUR, GBP, JPY)
        to_currency: Target currency code (USD, EUR, GBP, JPY)
    """
    # Mock exchange rates (1 USD = ...)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.50,
        "CAD": 1.35,
        "AUD": 1.52
    }
    
    if from_currency not in rates or to_currency not in rates:
        return f"Error: Unsupported currency. Supported: {', '.join(rates.keys())}"
    
    # Convert to USD first, then to target
    usd_amount = amount / rates[from_currency.upper()]
    target_amount = usd_amount * rates[to_currency.upper()]
    
    return f"{amount:.2f} {from_currency.upper()} = {target_amount:.2f} {to_currency.upper()}"

# Tool 5: Summarize Text
@tool
def summarize_text(text: str) -> str:
    """
    Summarize a piece of text.
    
    Args:
        text: The text to summarize
    """
    # Simple summarization (for demo purposes)
    words = text.split()
    if len(words) <= 20:
        return f"Text summary: {text}"
    
    # Take first 10 and last 10 words
    start = " ".join(words[:10])
    end = " ".join(words[-10:])
    summary = f"{start} ... {end} ({len(words)} words)"
    
    return f"Summary: {summary}"

# ============================================
# STEP 2: TOOL REGISTRY & SCHEMAS
# ============================================

TOOLS = {
    "search_db": search_db,
    "calculate": calculate,
    "format_date": format_date,
    "convert_currency": convert_currency,
    "summarize_text": summarize_text
}

def get_tool_schemas():
    """Return tool schemas for the LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_db",
                "description": "Search the knowledge database for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "format_date",
                "description": "Get the current date in a specific format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format_type": {
                            "type": "string",
                            "description": "Format type: 'short', 'long', 'iso', 'us'",
                            "enum": ["short", "long", "iso", "us", "european"]
                        }
                    },
                    "required": ["format_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert currency using mock exchange rates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "The amount to convert"},
                        "from_currency": {"type": "string", "description": "Source currency code (USD, EUR, GBP, JPY)"},
                        "to_currency": {"type": "string", "description": "Target currency code"}
                    },
                    "required": ["amount", "from_currency", "to_currency"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_text",
                "description": "Summarize a piece of text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to summarize"}
                    },
                    "required": ["text"]
                }
            }
        }
    ]

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name in TOOLS:
        try:
            # Convert args to the format the tool expects
            result = TOOLS[tool_name].invoke(**tool_args)
            return result
        except Exception as e:
            return f"Tool execution error: {e}"
    return f"Error: Unknown tool '{tool_name}'"

# ============================================
# STEP 3: ADVANCED TOOL-CALLING AGENT
# ============================================

class AdvancedToolAgent:
    """Agent with 5 tools."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        self.tool_schemas = get_tool_schemas()
        self.max_iterations = 5
        self.messages = [
            SystemMessage(content="""You are a helpful assistant with access to 5 tools.

Available tools:
1. search_db: Search for information (parameter: query)
2. calculate: Do math (parameter: expression)
3. format_date: Get current date in format (parameter: format_type)
4. convert_currency: Convert currency (parameters: amount, from_currency, to_currency)
5. summarize_text: Summarize text (parameter: text)

When you need to use a tool, call it with the required parameters.
If you don't need a tool, just answer directly.
""")
        ]
    
    def parse_tool_calls(self, response):
        """Extract tool calls from the LLM response."""
        tool_calls = []
        
        # Check for native tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.tool_calls
        
        # Parse from text (for models without native tool calling)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Pattern: [TOOL: tool_name: param1: value1, param2: value2]
        pattern = r'\[TOOL:\s*(\w+):\s*(.*?)\]'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        for match in matches:
            tool_name = match[0].strip().lower()
            args_str = match[1].strip()
            
            # Parse arguments (simple version)
            args = {}
            for part in args_str.split(','):
                if ':' in part:
                    key, value = part.split(':', 1)
                    args[key.strip()] = value.strip()
            
            # For tools with single string parameter
            if tool_name == "search_db":
                args = {"query": args_str if not args else args.get('query', args_str)}
            elif tool_name == "format_date":
                args = {"format_type": args_str if not args else args.get('format_type', 'short')}
            elif tool_name == "summarize_text":
                args = {"text": args_str if not args else args.get('text', args_str)}
            
            tool_calls.append({"name": tool_name, "args": args})
        
        return tool_calls
    
    def run(self, user_query: str):
        """Run the agent to answer a query."""
        print(f"\n🧑 User: {user_query}")
        print("-" * 50)
        
        self.messages.append(HumanMessage(content=user_query))
        iteration = 0
        final_answer = None
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            response = self.llm.invoke(
                self.messages,
                tools=self.tool_schemas
            )
            
            tool_calls = self.parse_tool_calls(response)
            
            if not tool_calls:
                final_answer = response.content
                print(f"✅ Final: {final_answer[:200]}...")
                break
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                print(f"🔧 Tool: {tool_name}({json.dumps(tool_args)})")
                result = execute_tool(tool_name, tool_args)
                print(f"📊 Result: {result}")
                
                self.messages.append(response)
                self.messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get("id", "1")
                ))
        
        if not final_answer:
            final_answer = "Could not complete the task."
        
        return final_answer

# ============================================
# STEP 4: TEST INDIVIDUAL TOOLS
# ============================================

def test_individual_tools():
    print("=" * 60)
    print("🧪 TESTING INDIVIDUAL TOOLS")
    print("=" * 60)
    
    # Test 1: search_db
    print("\n1️⃣ search_db:")
    print(search_db.invoke("rag"))
    print(search_db.invoke("llm"))
    print(search_db.invoke("unknown"))
    
    # Test 2: calculate
    print("\n2️⃣ calculate:")
    print(calculate.invoke("25 * 4"))
    print(calculate.invoke("10 + 20"))
    print(calculate.invoke("100 / 0"))
    
    # Test 3: format_date
    print("\n3️⃣ format_date:")
    print(format_date.invoke("short"))
    print(format_date.invoke("long"))
    print(format_date.invoke("us"))
    
    # Test 4: convert_currency
    print("\n4️⃣ convert_currency:")
    print(convert_currency.invoke(100, "USD", "EUR"))
    print(convert_currency.invoke(50, "GBP", "USD"))
    
    # Test 5: summarize_text
    print("\n5️⃣ summarize_text:")
    long_text = "This is a long piece of text about artificial intelligence. " * 5
    print(summarize_text.invoke(long_text))

# ============================================
# STEP 5: TEST TOGETHER
# ============================================

def test_together():
    print("\n" + "=" * 60)
    print("🤖 TESTING TOOLS TOGETHER")
    print("=" * 60)
    
    agent = AdvancedToolAgent()
    
    test_queries = [
        "What is RAG?",
        "Calculate 25 * 4",
        "What's today's date in long format?",
        "Convert 100 USD to EUR",
        "Summarize this: AI is transforming the world. It helps doctors, teachers, and engineers."
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        result = agent.run(query)
        print(f"\n📝 Final: {result}")
        print("=" * 60)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Test individual tools first
    test_individual_tools()
    
    # Then test together
    test_together()
