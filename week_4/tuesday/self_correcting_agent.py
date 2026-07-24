#!/usr/bin/env python3
"""
Week 4 - Tuesday: Self-Correcting Agent Loop
Generate → Validate → [Pass: Respond | Fail: Regenerate] (max 3x)
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

class AgentState(TypedDict):
    """State for the self-correcting agent."""
    
    # Input
    query: str
    
    # Generation
    current_response: str
    attempt_number: int
    max_attempts: int
    
    # Validation
    is_valid: bool
    validation_feedback: str
    
    # Result
    final_response: str
    status: str  # "pending", "valid", "invalid", "max_attempts"

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def generate_response(state: AgentState) -> AgentState:
    """Generate a response using Groq."""
    print(f"🤖 [GENERATE] Attempt {state['attempt_number'] + 1}...")
    
    query = state.get("query", "")
    attempt = state.get("attempt_number", 0)
    
    # If this is a retry, include feedback
    feedback = state.get("validation_feedback", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )
        
        if feedback and attempt > 0:
            prompt = f"""
            You previously answered this query, but it was rejected.
            
            Query: {query}
            
            Feedback: {feedback}
            
            Please provide a better response addressing the feedback.
            """
        else:
            prompt = f"Answer this question clearly and accurately:\n\n{query}"
        
        response = llm.invoke(prompt)
        
        state["current_response"] = response.content
        state["status"] = "generated"
        
        print(f"   Response length: {len(response.content)} characters")
        print(f"   Response preview: {response.content[:100]}...")
        
    except Exception as e:
        state["current_response"] = f"Error: {e}"
        state["status"] = "error"
    
    return state

def validate_response(state: AgentState) -> AgentState:
    """Validate the generated response."""
    print("🔍 [VALIDATE] Checking response quality...")
    
    response = state.get("current_response", "")
    query = state.get("query", "")
    
    # Basic validation checks
    issues = []
    
    # Check 1: Response is not empty
    if not response or len(response) < 10:
        issues.append("Response is too short or empty")
    
    # Check 2: Response actually addresses the query
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = query_words & response_words
    
    if len(overlap) < 2 and len(query_words) > 3:
        issues.append("Response doesn't seem to address the query")
    
    # Check 3: Response has substance (not just "I don't know")
    if any(phrase in response.lower() for phrase in ["i don't know", "i cannot", "i'm not sure"]):
        if len(response) < 50:
            issues.append("Response is too vague")
    
    # Check 4: Check for profanity or harmful content (simple check)
    harmful_words = ["kill", "destroy", "hack", "steal", "cheat"]
    if any(word in response.lower() for word in harmful_words):
        issues.append("Response contains potentially harmful content")
    
    if issues:
        state["is_valid"] = False
        state["validation_feedback"] = " ".join(issues)
        state["status"] = "invalid"
        print(f"   ❌ Invalid: {issues[0]}")
    else:
        state["is_valid"] = True
        state["status"] = "valid"
        print("   ✅ Valid response")
    
    return state

def respond_to_user(state: AgentState) -> AgentState:
    """Return the final valid response."""
    print("💬 [RESPOND] Returning validated response...")
    
    state["final_response"] = state.get("current_response", "")
    state["status"] = "complete"
    
    print(f"   Final response: {state['final_response'][:100]}...")
    
    return state

def regenerate_response(state: AgentState) -> AgentState:
    """Prepare for regeneration."""
    print("🔄 [REGENERATE] Preparing for retry...")
    
    attempt = state.get("attempt_number", 0) + 1
    
    if attempt >= state.get("max_attempts", 3):
        print("   ⚠️ Max attempts reached!")
        state["status"] = "max_attempts"
        state["final_response"] = f"Unable to generate a valid response after {attempt} attempts."
        return state
    
    state["attempt_number"] = attempt
    state["status"] = "retry"
    
    print(f"   Attempt {attempt + 1} of {state.get('max_attempts', 3)}")
    
    return state

# ============================================
# STEP 3: ROUTING FUNCTION
# ============================================

def should_continue(state: AgentState) -> Literal["respond", "regenerate", "fail"]:
    """Decide whether to respond, regenerate, or fail."""
    
    if state.get("status") == "max_attempts":
        print("   🛑 Max attempts reached, stopping")
        return "fail"
    
    if state.get("is_valid", False):
        print("   ✅ Valid response found, responding")
        return "respond"
    
    if state.get("attempt_number", 0) >= state.get("max_attempts", 3):
        print("   🛑 Attempt limit reached, failing")
        return "fail"
    
    print("   🔄 Regenerating...")
    return "regenerate"

# ============================================
# STEP 4: BUILD THE GRAPH
# ============================================

def build_self_correcting_graph(max_attempts: int = 3):
    """Build the self-correcting agent graph."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("generate", generate_response)
    graph.add_node("validate", validate_response)
    graph.add_node("respond", respond_to_user)
    graph.add_node("regenerate", regenerate_response)
    graph.add_node("fail", lambda state: state)  # Simple pass-through
    
    # Set entry point
    graph.set_entry_point("generate")
    
    # Flow: generate → validate
    graph.add_edge("generate", "validate")
    
    # Conditional: validate → respond or regenerate
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {
            "respond": "respond",
            "regenerate": "regenerate",
            "fail": "fail"
        }
    )
    
    # Regenerate goes back to generate
    graph.add_edge("regenerate", "generate")
    
    # Respond and fail end the graph
    graph.add_edge("respond", END)
    graph.add_edge("fail", END)
    
    # Set max attempts
    graph.compile()
    
    # Create a function that sets max_attempts in state
    def run_with_max(state):
        state["max_attempts"] = max_attempts
        return graph.invoke(state)
    
    return run_with_max

# ============================================
# STEP 5: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("🔄 SELF-CORRECTING AGENT LOOP")
    print("=" * 60)
    
    test_queries = [
        "Explain what RAG is.",
        "How do I hack a computer?",  # Should fail validation
        "What is LangChain?"
    ]
    
    run_agent = build_self_correcting_graph(max_attempts=3)
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"📝 Query: {query}")
        print("-" * 40)
        
        initial_state = {
            "query": query,
            "current_response": "",
            "attempt_number": 0,
            "max_attempts": 3,
            "is_valid": False,
            "validation_feedback": "",
            "final_response": "",
            "status": "pending"
        }
        
        final_state = run_agent(initial_state)
        
        print("\n📊 Results:")
        print(f"   Status: {final_state.get('status', 'unknown')}")
        print(f"   Attempts: {final_state.get('attempt_number', 0) + 1}")
        print(f"\n   Final Response:\n{final_state.get('final_response', 'No response')[:300]}...")

if __name__ == "__main__":
    run_demo()
