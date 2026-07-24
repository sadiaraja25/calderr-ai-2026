#!/usr/bin/env python3
"""
Lab 4.2: Self-Correcting Agent Loop
Concept: Tuesday - Branching & Loops + Wednesday - Stateful Agents
"""

import os
from typing import TypedDict, Literal, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: STATE
# ============================================

class AgentState(TypedDict):
    """State for self-correcting agent."""
    
    query: str
    current_response: str
    attempt: int
    max_attempts: int
    is_valid: bool
    feedback: str
    final_response: str
    history: List[str]
    status: str

# ============================================
# STEP 2: NODES
# ============================================

def generate_response(state: AgentState) -> AgentState:
    """Generate a response."""
    print(f"🤖 [GENERATE] Attempt {state.get('attempt', 0) + 1}...")
    
    query = state.get("query", "")
    feedback = state.get("feedback", "")
    attempt = state.get("attempt", 0)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
    
    if feedback and attempt > 0:
        prompt = f"""
        Previous answer was rejected.
        Query: {query}
        Feedback: {feedback}
        Provide a better answer addressing the feedback.
        """
    else:
        prompt = f"Answer this clearly: {query}"
    
    response = llm.invoke(prompt)
    
    state["current_response"] = response.content
    state["status"] = "generated"
    
    print(f"   Response: {response.content[:100]}...")
    return state

def validate_response(state: AgentState) -> AgentState:
    """Validate the response."""
    print("🔍 [VALIDATE] Checking response...")
    
    response = state.get("current_response", "")
    query = state.get("query", "")
    issues = []
    
    # Check 1: Not empty
    if len(response) < 10:
        issues.append("Response too short")
    
    # Check 2: Addresses query
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    if len(query_words & response_words) < 2:
        issues.append("Doesn't address query")
    
    # Check 3: Has substance
    vague = ["i don't know", "i cannot", "i'm not sure"]
    if any(phrase in response.lower() for phrase in vague):
        if len(response) < 50:
            issues.append("Response too vague")
    
    if issues:
        state["is_valid"] = False
        state["feedback"] = " ".join(issues)
        state["status"] = "invalid"
        print(f"   ❌ Issues: {issues[0]}")
    else:
        state["is_valid"] = True
        state["status"] = "valid"
        print("   ✅ Valid response")
    
    return state

def respond_to_user(state: AgentState) -> AgentState:
    """Return final response."""
    print("💬 [RESPOND] Returning response...")
    state["final_response"] = state.get("current_response", "")
    state["status"] = "complete"
    return state

def regenerate_response(state: AgentState) -> AgentState:
    """Prepare for retry."""
    print("🔄 [REGENERATE] Retrying...")
    
    attempt = state.get("attempt", 0) + 1
    state["attempt"] = attempt
    
    # Add to history
    history = state.get("history", [])
    history.append(f"Attempt {attempt}: {state.get('feedback', '')}")
    state["history"] = history
    
    if attempt >= state.get("max_attempts", 3):
        state["status"] = "max_attempts"
        print("   ⚠️ Max attempts reached")
    else:
        state["status"] = "retry"
        print(f"   Attempt {attempt + 1} of {state.get('max_attempts', 3)}")
    
    return state

# ============================================
# STEP 3: ROUTING
# ============================================

def should_continue(state: AgentState) -> Literal["respond", "regenerate", "fail"]:
    """Decide next step."""
    
    if state.get("status") == "max_attempts":
        return "fail"
    
    if state.get("is_valid", False):
        return "respond"
    
    if state.get("attempt", 0) >= state.get("max_attempts", 3):
        return "fail"
    
    return "regenerate"

# ============================================
# STEP 4: BUILD GRAPH
# ============================================

def build_graph():
    """Build the self-correcting graph."""
    
    graph = StateGraph(AgentState)
    
    graph.add_node("generate", generate_response)
    graph.add_node("validate", validate_response)
    graph.add_node("respond", respond_to_user)
    graph.add_node("regenerate", regenerate_response)
    
    graph.set_entry_point("generate")
    
    graph.add_edge("generate", "validate")
    
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {
            "respond": "respond",
            "regenerate": "regenerate",
            "fail": END
        }
    )
    
    graph.add_edge("regenerate", "generate")
    graph.add_edge("respond", END)
    
    return graph.compile()

# ============================================
# STEP 5: TEST
# ============================================

def test_graph():
    print("=" * 60)
    print("🧪 Lab 4.2: Self-Correcting Agent Loop")
    print("=" * 60)
    
    graph = build_graph()
    
    test_queries = [
        "What is RAG?",
        "Explain AI in simple terms",
        "How does machine learning work?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        state = {
            "query": query,
            "current_response": "",
            "attempt": 0,
            "max_attempts": 3,
            "is_valid": False,
            "feedback": "",
            "final_response": "",
            "history": [],
            "status": "pending"
        }
        
        result = graph.invoke(state)
        
        print(f"\n✅ Result:")
        print(f"   Status: {result['status']}")
        print(f"   Attempts: {result['attempt'] + 1}")
        print(f"   History: {len(result.get('history', []))} retries")
        print(f"\n   Response: {result['final_response'][:200]}...")

if __name__ == "__main__":
    test_graph()
