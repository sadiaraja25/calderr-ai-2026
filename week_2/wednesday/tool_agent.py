#!/usr/bin/env python3
"""
Week 2 - Wednesday: Tool Calling Basics
Build a tool-calling agent with weather and calculator tools.
"""

import os
import json
import re
from pathlib import Path
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
# STEP 1: DEFINE TOOLS (Weather + Calculator)
# ============================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The name of the city (e.g., "London", "Paris")
    """
    weather_data = {
        "london": "15°C, sunny",
        "paris": "18°C, cloudy",
        "new york": "22°C, clear",
        "tokyo": "25°C, humid",
        "sydney": "20°C, partly cloudy"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: Math expression (e.g., "2 + 2", "25 * 4")
    """
    try:
        # Validate input
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        
        # Calculate
        result = eval(expression)
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {e}"

# ============================================
# STEP 2: CREATE TOOL SCHEMAS (for the LLM)
# ============================================

def get_tool_schemas():
    """Return the tool schemas for the LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to calculate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

# ============================================
# STEP 3: TOOL EXECUTOR
# ============================================

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name == "get_weather":
        city = tool_args.get("city", "")
        return get_weather.invoke(city)
    elif tool_name == "calculator":
        expression = tool_args.get("expression", "")
        return calculator.invoke(expression)
    else:
        return f"Error: Unknown tool '{tool_name}'"

# ============================================
# STEP 4: TOOL-CALLING AGENT
# ============================================

class ToolCallingAgent:
    """Agent that can call tools to answer questions."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        self.tool_schemas = get_tool_schemas()
        self.max_iterations = 5
        self.messages = [
            SystemMessage(content="""You are a helpful assistant with access to tools.
            
Available tools:
- get_weather: Get weather for a city (parameter: city)
- calculator: Calculate math expressions (parameter: expression)

When you need to use a tool, call it with the required parameters.
If you don't need a tool, just answer directly.
""")
        ]
    
    def parse_tool_calls(self, response):
        """Extract tool calls from the LLM response."""
        tool_calls = []
        
        # Check if the response contains tool call information
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.tool_calls
        
        # For models that don't support native tool calling, we parse from text
        # Look for patterns like: [TOOL: get_weather: London]
        content = response.content if hasattr(response, 'content') else str(response)
        pattern = r'\[TOOL:\s*(\w+):\s*(.*?)\]'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        for match in matches:
            tool_name = match[0].strip().lower()
            tool_input = match[1].strip()
            
            # Parse arguments (for simplicity, we treat everything as a string)
            if tool_name == "get_weather":
                args = {"city": tool_input}
            elif tool_name == "calculator":
                args = {"expression": tool_input}
            else:
                continue
            
            tool_calls.append({
                "name": tool_name,
                "args": args
            })
        
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
            
            # Get LLM response
            response = self.llm.invoke(
                self.messages,
                tools=self.tool_schemas
            )
            
            # Check for tool calls
            tool_calls = self.parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, this is the final answer
                final_answer = response.content
                print(f"✅ Final Answer: {final_answer[:200]}...")
                break
            
            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                print(f"🔧 Tool Call: {tool_name}({json.dumps(tool_args)})")
                
                # Execute the tool
                result = execute_tool(tool_name, tool_args)
                print(f"📊 Tool Result: {result}")
                
                # Add the tool result to the conversation
                self.messages.append(response)
                self.messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call.get("id", "1")
                ))
        
        if not final_answer:
            final_answer = "Could not complete the task."
        
        return final_answer

# ============================================
# STEP 5: TEST THE AGENT
# ============================================

def run_tests():
    print("=" * 60)
    print("🛠️ TOOL-CALLING AGENT (Weather + Calculator)")
    print("=" * 60)
    
    agent = ToolCallingAgent()
    
    # Test queries
    test_queries = [
        "What's the weather in London?",
        "Calculate 25 * 4",
        "What's the weather in Paris and calculate 100 / 2?",
        "What is 15 + 7?",
        "What's the weather in Tokyo?"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        result = agent.run(query)
        print(f"\n📝 Final: {result}")
        print("=" * 60)

if __name__ == "__main__":
    run_tests()
