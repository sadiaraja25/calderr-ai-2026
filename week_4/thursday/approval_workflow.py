#!/usr/bin/env python3
"""
Week 4 - Thursday: Simple Approval Workflow
Agent proposes action → Human reviews → Approves/Rejects → Agent adjusts
"""

import os
import json
from typing import TypedDict, Literal, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class ApprovalState(TypedDict):
    """State for the approval workflow."""
    
    # Input
    user_request: str
    
    # Agent's proposal
    proposed_action: str
    proposed_details: dict
    
    # Human review
    human_decision: Optional[str]  # "approved", "rejected", "needs_revision"
    human_feedback: Optional[str]
    
    # Result
    final_action: str
    status: str
    iteration_count: int

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def propose_action(state: ApprovalState) -> ApprovalState:
    """Agent proposes an action."""
    print("🤖 [PROPOSE] Agent generating proposal...")
    
    request = state.get("user_request", "")
    iteration = state.get("iteration_count", 0)
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        User Request: {request}
        
        Propose a specific action to fulfill this request.
        Include:
        1. What action to take
        2. Why this action is appropriate
        
        Action:"""
        
        response = llm.invoke(prompt)
        
        state["proposed_action"] = response.content
        state["proposed_details"] = {"request": request, "iteration": iteration}
        state["status"] = "proposed"
        
        print(f"   Proposal: {response.content[:100]}...")
        
    except Exception as e:
        state["proposed_action"] = f"Error generating proposal: {e}"
        state["status"] = "error"
    
    return state

def human_review(state: ApprovalState) -> ApprovalState:
    """
    Human reviews the proposal (this is where the interrupt happens).
    In production, this would be a separate UI.
    """
    print("👤 [REVIEW] Human review required!")
    print("-" * 40)
    print(f"Proposal: {state.get('proposed_action', 'No proposal')}")
    print("-" * 40)
    
    print("\nOptions:")
    print("  1. Approve")
    print("  2. Reject")
    print("  3. Needs Revision")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        state["human_decision"] = "approved"
        state["human_feedback"] = "Approved by human reviewer"
        print("   ✅ Approved")
    elif choice == "2":
        state["human_decision"] = "rejected"
        state["human_feedback"] = input("Reason for rejection: ")
        print(f"   ❌ Rejected: {state['human_feedback']}")
    elif choice == "3":
        state["human_decision"] = "needs_revision"
        state["human_feedback"] = input("Revision feedback: ")
        print(f"   🔄 Revision requested: {state['human_feedback']}")
    else:
        state["human_decision"] = "needs_revision"
        state["human_feedback"] = "Please revise the proposal"
        print("   🔄 Default: Revision requested")
    
    state["status"] = "reviewed"
    return state

def execute_action(state: ApprovalState) -> ApprovalState:
    """Execute the approved action."""
    print("⚡ [EXECUTE] Executing approved action...")
    
    state["final_action"] = state.get("proposed_action", "")
    state["status"] = "executed"
    
    print(f"   Executed: {state['final_action'][:100]}...")
    return state

def revise_proposal(state: ApprovalState) -> ApprovalState:
    """Revise the proposal based on human feedback."""
    print("🔄 [REVISE] Revising proposal based on feedback...")
    
    feedback = state.get("human_feedback", "No feedback provided")
    previous_proposal = state.get("proposed_action", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        Revise this proposal based on feedback.
        
        Previous Proposal:
        {previous_proposal}
        
        Feedback:
        {feedback}
        
        Revised Proposal:"""
        
        response = llm.invoke(prompt)
        
        state["proposed_action"] = response.content
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["status"] = "revised"
        
        print(f"   Revised proposal: {response.content[:100]}...")
        
    except Exception as e:
        state["proposed_action"] = f"Error revising: {e}"
        state["status"] = "error"
    
    return state

# ============================================
# STEP 3: ROUTING FUNCTION
# ============================================

def route_after_review(state: ApprovalState) -> Literal["execute", "revise"]:
    """Route based on human decision."""
    
    decision = state.get("human_decision", "needs_revision")
    
    if decision == "approved":
        print("   ✅ Proceeding to execute")
        return "execute"
    else:
        print("   🔄 Routing to revision")
        return "revise"

# ============================================
# STEP 4: BUILD THE GRAPH (FIXED)
# ============================================

def build_approval_graph():
    """Build the approval workflow graph with checkpointer."""
    
    graph = StateGraph(ApprovalState)
    
    # Add nodes
    graph.add_node("propose", propose_action)
    graph.add_node("review", human_review)
    graph.add_node("execute", execute_action)
    graph.add_node("revise", revise_proposal)
    
    # Set entry point
    graph.set_entry_point("propose")
    
    # Add edges
    graph.add_edge("propose", "review")
    
    # Conditional: review → execute OR revise
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "execute": "execute",
            "revise": "revise"
        }
    )
    
    # Revise goes back to review
    graph.add_edge("revise", "review")
    
    # Execute ends
    graph.add_edge("execute", END)
    
    # ✅ FIXED: Compile with checkpointer here
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

# ============================================
# STEP 5: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("🧑‍💼 APPROVAL WORKFLOW DEMO")
    print("=" * 60)
    
    # ✅ FIXED: The graph already has the checkpointer
    graph = build_approval_graph()
    
    # Thread ID for this conversation
    thread_id = "approval_1"
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "user_request": "I want to delete all customer data from our database",
        "proposed_action": "",
        "proposed_details": {},
        "human_decision": None,
        "human_feedback": None,
        "final_action": "",
        "status": "pending",
        "iteration_count": 0
    }
    
    print("\n📝 Initial State:")
    print(f"   User Request: {initial_state['user_request']}")
    
    print("\n" + "=" * 60)
    print("🔄 Running Graph (with Human-in-the-Loop)")
    print("=" * 60)
    
    # Run the graph
    final_state = graph.invoke(initial_state, config)
    
    print("\n" + "=" * 60)
    print("📊 Final State")
    print("=" * 60)
    print(f"   Status: {final_state.get('status', 'unknown')}")
    print(f"   Iterations: {final_state.get('iteration_count', 0)}")
    print(f"   Human Decision: {final_state.get('human_decision', 'none')}")
    print(f"\n   Final Action:\n{final_state.get('final_action', 'No action')[:200]}...")

if __name__ == "__main__":
    run_demo()
