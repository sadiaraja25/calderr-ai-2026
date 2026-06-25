# ============================================
# react_agent.py - Manual ReAct Implementation
# ============================================

# TOOL 1: Mock Database
mock_db = {
    "france": "France has 68 million people.",
    "paris": "Paris is the capital of France.",
    "germany": "Germany has 84 million people.",
    "berlin": "Berlin is the capital of Germany.",
    "population": "The world population is 8 billion.",
}

def search_mock(query):
    """Search the mock database."""
    query_lower = query.lower()
    results = []
    for key, value in mock_db.items():
        if key in query_lower:
            results.append(value)
    return " ".join(results) if results else f"No info found about '{query}'."

# TOOL 2: Calculator
def calculate(expression):
    """Perform math calculations."""
    try:
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Invalid expression."
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Math error: {e}"

# TOOL 3: Default
def default_answer(query):
    """Default response."""
    return f"I don't know about '{query}'. Try math or topics like France, Germany."

# ReAct Agent
def decide_tool(query):
    """Reason: Decide which tool to use."""
    query_lower = query.lower()
    
    if any(op in query for op in ["+", "-", "*", "/"]):
        return "calculate", query
    
    for key in mock_db.keys():
        if key in query_lower:
            return "search", query
    
    return "default", query

def react_agent(query):
    """Main ReAct loop."""
    print(f"\n🧑 You: {query}")
    print("🤔 Reasoning...")
    
    # Reason
    tool, tool_input = decide_tool(query)
    print(f"🔧 Tool: {tool}")
    
    # Act
    if tool == "search":
        result = search_mock(tool_input)
    elif tool == "calculate":
        result = calculate(tool_input)
    else:
        result = default_answer(query)
    
    # Observe & Respond
    print(f"📊 Result: {result}")
    print(f"🤖 Agent: {result}")
    return result

# Main Loop
print("=" * 50)
print("🤖 Manual ReAct Agent")
print("=" * 50)
print("Try: 'What is Paris?', 'Calculate 5+3', 'Tell me about Germany'")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("\n❓ You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break
    if user_input:
        react_agent(user_input)
