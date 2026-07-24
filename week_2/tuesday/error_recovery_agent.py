#!/usr/bin/env python3
"""
Week 2 - Tuesday: Lab 2.3 - Error Recovery Agent
Handles tool failures with alternative strategies.
"""

import os
import time
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

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
# STEP 1: TOOLS WITH FAILURE MODES
# ============================================

class Tools:
    """Collection of tools with failure modes for testing."""
    
    @staticmethod
    def unreliable_search(query: str) -> str:
        """Search with random failures."""
        # Simulate rate limiting errors
        if random.random() < 0.4:
            return "ERROR: Rate limit exceeded. Please try again."
        
        knowledge_base = {
            "rag": "RAG stands for Retrieval-Augmented Generation",
            "llm": "LLM stands for Large Language Model",
            "agent": "An agent uses LLMs with tools"
        }
        
        for key, value in knowledge_base.items():
            if key in query.lower():
                return f"Search result: {value}"
        return "Search result: No information found"
    
    @staticmethod
    def backup_search(query: str) -> str:
        """Backup search with different data."""
        backup_db = {
            "rag": "RAG combines retrieval and generation",
            "llm": "LLM processes language with transformers",
            "agent": "Agents can use multiple tools"
        }
        
        for key, value in backup_db.items():
            if key in query.lower():
                return f"Backup result: {value}"
        return "Backup result: No information found"
    
    @staticmethod
    def calculate(expression: str) -> str:
        """Calculate with potential syntax errors."""
        try:
            allowed = "0123456789+-*/(). "
            if not all(c in allowed for c in expression):
                return "ERROR: Invalid characters"
            return f"Result: {eval(expression)}"
        except Exception as e:
            return f"ERROR: {e}"

# ============================================
# STEP 2: ERROR RECOVERY AGENT
# ============================================

class ErrorRecoveryAgent:
    def __init__(self):
        self.max_attempts = 3
        self.backoff_seconds = 1
        self.max_backoff = 32
        self.logs = []
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
    
    def log(self, message):
        """Log agent activity."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
    
    def execute_with_retry(self, tool_func, tool_input, tool_name):
        """Execute a tool with exponential backoff."""
        attempt = 0
        last_error = None
        
        while attempt < self.max_attempts:
            attempt += 1
            self.log(f"Attempt {attempt} for {tool_name}")
            
            try:
                result = tool_func(tool_input)
                
                if "ERROR:" in result:
                    # Handle rate limit errors specifically
                    if "Rate limit" in result:
                        backoff = min(self.backoff_seconds * (2 ** (attempt - 1)), self.max_backoff)
                        self.log(f"Rate limited. Waiting {backoff} seconds...")
                        time.sleep(backoff)
                        continue
                    else:
                        # Other errors, try backup
                        self.log(f"Tool error: {result}")
                        return self.try_alternative(tool_name, tool_input)
                
                self.log(f"Success! Tool returned: {result[:100]}")
                return result
            
            except Exception as e:
                last_error = str(e)
                self.log(f"Exception: {e}")
                time.sleep(min(self.backoff_seconds * (2 ** (attempt - 1)), self.max_backoff))
        
        self.log(f"All attempts failed. Last error: {last_error}")
        return f"ERROR: All attempts failed - {last_error}"
    
    def try_alternative(self, tool_name, tool_input):
        """Try alternative approach for failed tools."""
        self.log(f"Trying alternative for {tool_name}")
        
        if tool_name == "search":
            # Use backup search
            self.log("Using backup search...")
            return Tools.backup_search(tool_input)
        else:
            # Default fallback
            return f"Alternative not available for {tool_name}"
    
    def run(self, user_query):
        """Run the error recovery agent."""
        self.log(f"Starting query: {user_query}")
        
        # For simplicity, we'll handle specific queries
        query_lower = user_query.lower()
        
        if "search" in query_lower or "rag" in query_lower or "llm" in query_lower:
            self.log("Routing to search tool...")
            result = self.execute_with_retry(
                Tools.unreliable_search,
                query_lower,
                "search"
            )
            return result
        
        elif "calculate" in query_lower or any(op in query_lower for op in ["+", "-", "*", "/"]):
            self.log("Routing to calculator tool...")
            # Extract expression
            import re
            numbers = re.findall(r'[\d\s\+\-\*\/\(\)\.]+', query_lower)
            expr = ''.join(numbers).strip() if numbers else "2+2"
            result = self.execute_with_retry(
                Tools.calculate,
                expr,
                "calculate"
            )
            return result
        
        else:
            return "I don't know how to handle that query."

# ============================================
# STEP 3: TEST THE AGENT
# ============================================

def run_tests():
    print("=" * 60)
    print("🔄 LAB 2.3: ERROR RECOVERY AGENT")
    print("=" * 60)
    
    agent = ErrorRecoveryAgent()
    
    test_queries = [
        "search for RAG",
        "calculate 25 * 4",
        "search for LLM",
        "calculate 100 / 0"  # This will cause division by zero error
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("-" * 40)
        
        result = agent.run(query)
        print(f"\n✅ Final Result: {result}")
        
        print("\n📋 Agent Logs:")
        for log in agent.logs[-3:]:
            print(f"  {log}")
        print("=" * 60)
        
        # Reset logs for next test
        agent.logs = []

if __name__ == "__main__":
    run_tests()
