#!/usr/bin/env python3
"""
Lab 4.3: Human-in-the-Loop Approval Workflow
Concept: Thursday - Human-in-the-Loop
"""

import os
import time
from typing import TypedDict, Literal, Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================
# STEP 1: STATE
# ============================================

class ModerationState(TypedDict):
    """State for content moderation."""
    
    post_id: str
    post_content: str
    author: str
    
    auto_score: float
    auto_decision: str  # approved, rejected, needs_review
    auto_reason: str
    
    human_decision: Optional[str]  # approved, rejected
    human_comment: Optional[str]
    
    final_decision: str
    final_reason: str
    status: str
    moderation_history: List[dict]

# ============================================
# STEP 2: NODES
# ============================================

def auto_moderate(state: ModerationState) -> ModerationState:
    """Auto-moderate content."""
    print("🤖 [AUTO] Auto-moderating...")
    
    content = state.get("post_content", "")
    
    # Simple risk scoring
    risky_keywords = ["hate", "violence", "spam", "scam"]
    risk_score = sum(1 for word in risky_keywords if word in content.lower()) / 4.0
    risk_score = min(risk_score, 1.0)
    
    state["auto_score"] = risk_score
    
    if risk_score < 0.3:
        state["auto_decision"] = "approved"
        state["auto_reason"] = "Content appears safe"
        print(f"   ✅ Auto-approved (score: {risk_score:.2f})")
    elif risk_score > 0.7:
        state["auto_decision"] = "rejected"
        state["auto_reason"] = "Content contains concerning keywords"
        print(f"   ❌ Auto-rejected (score: {risk_score:.2f})")
    else:
        state["auto_decision"] = "needs_review"
        state["auto_reason"] = f"Borderline content (score: {risk_score:.2f})"
        print(f"   ⚠️ Needs human review (score: {risk_score:.2f})")
    
    state["status"] = "auto_moderated"
    return state

def human_review(state: ModerationState) -> ModerationState:
    """Human review - INTERRUPT happens here."""
    print("👤 [HUMAN] Human review required!")
    print("-" * 40)
    print(f"Post: {state.get('post_content', '')}")
    print(f"Score: {state.get('auto_score', 0):.2f}")
    print(f"Reason: {state.get('auto_reason', '')}")
    print("-" * 40)
    
    print("\n1. Approve")
    print("2. Reject")
    
    choice = input("Enter choice (1/2): ")
    
    if choice == "1":
        state["human_decision"] = "approved"
        state["human_comment"] = "Approved by human"
        print("   ✅ Approved")
    else:
        state["human_decision"] = "rejected"
        state["human_comment"] = input("Reason: ")
        print(f"   ❌ Rejected: {state['human_comment']}")
    
    state["status"] = "human_reviewed"
    return state

def make_final_decision(state: ModerationState) -> ModerationState:
    """Make final decision."""
    print("✅ [FINAL] Making final decision...")
    
    auto_decision = state.get("auto_decision", "")
    human_decision = state.get("human_decision", "")
    
    if auto_decision == "approved":
        state["final_decision"] = "approved"
        state["final_reason"] = "Auto-approved"
    elif auto_decision == "rejected":
        state["final_decision"] = "rejected"
        state["final_reason"] = "Auto-rejected"
    else:
        if human_decision == "approved":
            state["final_decision"] = "approved"
            state["final_reason"] = f"Human approved: {state.get('human_comment', '')}"
        else:
            state["final_decision"] = "rejected"
            state["final_reason"] = f"Human rejected: {state.get('human_comment', '')}"
    
    # Add to history
    history = state.get("moderation_history", [])
    history.append({
        "timestamp": time.time(),
        "auto_score": state.get("auto_score", 0),
        "human_decision": state.get("human_decision", ""),
        "final_decision": state["final_decision"],
        "final_reason": state["final_reason"]
    })
    state["moderation_history"] = history
    
    state["status"] = "finalized"
    print(f"   Decision: {state['final_decision']}")
    return state

# ============================================
# STEP 3: ROUTING
# ============================================

def route_after_auto(state: ModerationState) -> Literal["final", "human"]:
    """Route after auto-moderation."""
    if state.get("auto_decision") in ["approved", "rejected"]:
        return "final"
    return "human"

# ============================================
# STEP 4: BUILD GRAPH (FIXED)
# ============================================

def build_graph(checkpointer=None):
    """
    Build moderation graph with optional persistence.
    
    Args:
        checkpointer: Optional checkpointer (MemorySaver or SqliteSaver)
    """
    graph = StateGraph(ModerationState)
    
    graph.add_node("auto", auto_moderate)
    graph.add_node("human", human_review)
    graph.add_node("final", make_final_decision)
    
    graph.set_entry_point("auto")
    
    graph.add_conditional_edges(
        "auto",
        route_after_auto,
        {
            "final": "final",
            "human": "human"
        }
    )
    
    graph.add_edge("human", "final")
    graph.add_edge("final", END)
    
    # Compile ONCE with checkpointer if provided
    return graph.compile(checkpointer=checkpointer)

# ============================================
# STEP 5: TEST WITH PERSISTENCE
# ============================================

def test_graph():
    print("=" * 60)
    print("🧪 Lab 4.3: Human-in-the-Loop Approval Workflow")
    print("=" * 60)
    
    # Create checkpointer for persistence
    checkpointer = MemorySaver()
    
    # Build graph WITH checkpointer in one step
    graph_with_persistence = build_graph(checkpointer=checkpointer)
    
    test_posts = [
        ("Post 1", "I love this community!", "user1"),
        ("Post 2", "This is spam spam spam!", "user2"),
        ("Post 3", "This is borderline content.", "user3"),
    ]
    
    for post_id, content, author in test_posts:
        thread_id = f"mod_{post_id.replace(' ', '_')}"
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n📝 {post_id}: {content[:50]}...")
        print("-" * 40)
        
        state = {
            "post_id": post_id,
            "post_content": content,
            "author": author,
            "auto_score": 0.0,
            "auto_decision": "",
            "auto_reason": "",
            "human_decision": None,
            "human_comment": None,
            "final_decision": "",
            "final_reason": "",
            "status": "pending",
            "moderation_history": []
        }
        
        result = graph_with_persistence.invoke(state, config)
        
        print(f"\n📊 Decision: {result['final_decision']}")
        print(f"   Reason: {result['final_reason']}")
        print(f"   History: {len(result.get('moderation_history', []))} entries")

if __name__ == "__main__":
    test_graph()

