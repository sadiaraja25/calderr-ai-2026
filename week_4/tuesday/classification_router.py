#!/usr/bin/env python3
"""
Week 4 - Tuesday: Classification Router
User query → [general/technical/sensitive] → specialized handler
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class RouterState(TypedDict):
    """State for the classification router."""
    
    # Input
    user_query: str
    
    # Classification
    category: str  # "general", "technical", "sensitive", "unknown"
    confidence: float
    
    # Result
    response: str
    handler_used: str

# ============================================
# STEP 2: CLASSIFICATION NODE
# ============================================

def classify_query(state: RouterState) -> RouterState:
    """
    Classify the user query into a category.
    """
    print("🔍 [CLASSIFY] Analyzing query...")
    
    query = state.get("user_query", "")
    
    # Keywords for classification
    technical_keywords = [
        "code", "python", "function", "algorithm", "api", "database",
        "debug", "error", "programming", "langchain", "rag"
    ]
    
    sensitive_keywords = [
        "password", "credit card", "ssn", "bank account", "personal",
        "confidential", "private", "secret"
    ]
    
    query_lower = query.lower()
    
    # Check for sensitive first (priority)
    if any(keyword in query_lower for keyword in sensitive_keywords):
        category = "sensitive"
        confidence = 0.9
        print("   ⚠️ Sensitive query detected")
    
    # Check for technical
    elif any(keyword in query_lower for keyword in technical_keywords):
        category = "technical"
        confidence = 0.85
        print("   💻 Technical query detected")
    
    # Use LLM for better classification (if ambiguous)
    elif len(query) > 20:
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                api_key=GROQ_API_KEY
            )
            
            prompt = f"""
            Classify this user query into ONE category: general, technical, or sensitive.
            
            Query: {query}
            
            Category:"""
            
            response = llm.invoke(prompt)
            category = response.content.strip().lower()
            
            # Clean up LLM response
            if "technical" in category:
                category = "technical"
            elif "sensitive" in category:
                category = "sensitive"
            elif "general" in category:
                category = "general"
            else:
                category = "general"
            
            confidence = 0.7
            print(f"   🤖 LLM classified as: {category}")
            
        except Exception as e:
            category = "general"
            confidence = 0.5
            print(f"   ⚠️ LLM failed, using default: general")
    
    else:
        category = "general"
        confidence = 0.6
        print("   📝 Short query, defaulting to: general")
    
    state["category"] = category
    state["confidence"] = confidence
    
    return state

# ============================================
# STEP 3: SPECIALIZED HANDLERS
# ============================================

def general_handler(state: RouterState) -> RouterState:
    """Handle general queries."""
    print("🌐 [GENERAL] Processing general query...")
    
    query = state.get("user_query", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"Answer this question in a helpful, general way:\n\n{query}"
        response = llm.invoke(prompt)
        
        state["response"] = response.content
        state["handler_used"] = "general"
        
        print(f"   Response: {response.content[:100]}...")
        
    except Exception as e:
        state["response"] = f"Error: {e}"
        state["handler_used"] = "general_error"
    
    return state

def technical_handler(state: RouterState) -> RouterState:
    """Handle technical queries with code."""
    print("💻 [TECHNICAL] Processing technical query...")
    
    query = state.get("user_query", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        You are a technical expert. Provide a detailed, code-focused answer.
        Include code examples where appropriate.
        
        Query: {query}
        """
        
        response = llm.invoke(prompt)
        
        state["response"] = response.content
        state["handler_used"] = "technical"
        
        print(f"   Response: {response.content[:100]}...")
        
    except Exception as e:
        state["response"] = f"Error: {e}"
        state["handler_used"] = "technical_error"
    
    return state

def sensitive_handler(state: RouterState) -> RouterState:
    """Handle sensitive queries with caution."""
    print("🔒 [SENSITIVE] Processing sensitive query...")
    
    query = state.get("user_query", "")
    
    # For sensitive queries, we don't use LLM directly
    # Instead, we provide a safe response
    state["response"] = """
    I notice your question may involve sensitive information. 
    For privacy and security reasons, I cannot process this query. 
    Please contact the appropriate support team for assistance.
    """
    state["handler_used"] = "sensitive"
    
    print(f"   Response: {state['response'][:100]}...")
    
    return state

def unknown_handler(state: RouterState) -> RouterState:
    """Handle unknown or ambiguous queries."""
    print("❓ [UNKNOWN] Processing unknown query...")
    
    state["response"] = """
    I'm not sure how to best handle your query. 
    Could you rephrase it or provide more context?
    """
    state["handler_used"] = "unknown"
    
    print(f"   Response: {state['response'][:100]}...")
    
    return state

# ============================================
# STEP 4: ROUTING FUNCTION
# ============================================

def route_query(state: RouterState) -> Literal["general", "technical", "sensitive", "unknown"]:
    """Route to the appropriate handler based on classification."""
    category = state.get("category", "general")
    
    print(f"   🚀 Routing to: {category}")
    
    if category == "general":
        return "general"
    elif category == "technical":
        return "technical"
    elif category == "sensitive":
        return "sensitive"
    else:
        return "unknown"

# ============================================
# STEP 5: BUILD THE GRAPH
# ============================================

def build_router_graph():
    """Build the classification router graph."""
    
    graph = StateGraph(RouterState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("general", general_handler)
    graph.add_node("technical", technical_handler)
    graph.add_node("sensitive", sensitive_handler)
    graph.add_node("unknown", unknown_handler)
    
    # Set entry point
    graph.set_entry_point("classify")
    
    # Conditional routing from classify
    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "general": "general",
            "technical": "technical",
            "sensitive": "sensitive",
            "unknown": "unknown"
        }
    )
    
    # All handlers go to END
    graph.add_edge("general", END)
    graph.add_edge("technical", END)
    graph.add_edge("sensitive", END)
    graph.add_edge("unknown", END)
    
    return graph.compile()

# ============================================
# STEP 6: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("🌿 CLASSIFICATION ROUTER")
    print("=" * 60)
    
    test_queries = [
        "What is the weather like today?",
        "How do I write a Python function to sort a list?",
        "What is my password for the system?",
        "Tell me about artificial intelligence."
    ]
    
    graph = build_router_graph()
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"📝 Query: {query}")
        print("-" * 40)
        
        initial_state = {
            "user_query": query,
            "category": "",
            "confidence": 0.0,
            "response": "",
            "handler_used": ""
        }
        
        final_state = graph.invoke(initial_state)
        
        print("\n📊 Results:")
        print(f"   Category: {final_state.get('category', 'unknown')}")
        print(f"   Handler: {final_state.get('handler_used', 'unknown')}")
        print(f"\n   Response: {final_state.get('response', 'No response')[:200]}...")

if __name__ == "__main__":
    run_demo()
