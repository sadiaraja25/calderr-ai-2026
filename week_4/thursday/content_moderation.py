#!/usr/bin/env python3
"""
Week 4 - Thursday: Content Moderation Graph
Posts flow through: auto-approve → borderline human review → final decision
"""

import os
import json
import time
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

class ModerationState(TypedDict):
    """State for the content moderation graph."""
    
    # Input
    post_id: str
    post_content: str
    author_id: str
    
    # Auto-moderation
    auto_score: float  # 0-1, higher = more risky
    auto_reason: str
    auto_decision: str  # "approved", "rejected", "needs_review"
    
    # Human review
    human_decision: Optional[str]  # "approved", "rejected"
    human_reviewer: Optional[str]
    human_comment: Optional[str]
    
    # Final
    final_decision: str
    final_reason: str
    status: str
    moderation_history: List[dict]

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def auto_moderate(state: ModerationState) -> ModerationState:
    """Auto-moderate the content."""
    print("🤖 [AUTO] Auto-moderating content...")
    
    post_content = state.get("post_content", "")
    
    # Simulate auto-moderation
    risky_keywords = ["hate", "violence", "spam", "scam", "fraud"]
    content_lower = post_content.lower()
    risky_count = sum(1 for word in risky_keywords if word in content_lower)
    
    # Calculate risk score
    risk_score = min(risky_count / 3, 1.0)
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
    """Human review for borderline content."""
    print("👤 [HUMAN] Human review required!")
    print("-" * 40)
    print(f"Post: {state.get('post_content', 'No content')}")
    print(f"Auto-score: {state.get('auto_score', 0):.2f}")
    print(f"Reason: {state.get('auto_reason', 'No reason')}")
    print("-" * 40)
    
    print("\nOptions:")
    print("  1. Approve")
    print("  2. Reject")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == "1":
        state["human_decision"] = "approved"
        state["human_comment"] = "Human reviewer approved the content"
        print("   ✅ Human approved")
    else:
        state["human_decision"] = "rejected"
        state["human_comment"] = input("Reason for rejection: ")
        print(f"   ❌ Human rejected: {state['human_comment']}")
    
    state["human_reviewer"] = "human_reviewer_1"
    state["status"] = "human_reviewed"
    
    return state

def make_final_decision(state: ModerationState) -> ModerationState:
    """Make final decision based on all reviews."""
    print("✅ [FINAL] Making final decision...")
    
    auto_decision = state.get("auto_decision", "")
    human_decision = state.get("human_decision", "")
    
    if auto_decision == "approved":
        state["final_decision"] = "approved"
        state["final_reason"] = "Auto-approved by system"
    elif auto_decision == "rejected":
        state["final_decision"] = "rejected"
        state["final_reason"] = "Auto-rejected by system"
    else:
        # Borderline case - use human decision
        if human_decision == "approved":
            state["final_decision"] = "approved"
            state["final_reason"] = f"Approved by human reviewer: {state.get('human_comment', '')}"
        elif human_decision == "rejected":
            state["final_decision"] = "rejected"
            state["final_reason"] = f"Rejected by human reviewer: {state.get('human_comment', '')}"
        else:
            # Fallback - reject if no human decision
            state["final_decision"] = "rejected"
            state["final_reason"] = "No human decision provided"
    
    # Add to history
    history = state.get("moderation_history", [])
    history.append({
        "timestamp": time.time(),
        "auto_score": state.get("auto_score", 0),
        "auto_decision": state.get("auto_decision", ""),
        "human_decision": state.get("human_decision", ""),
        "final_decision": state["final_decision"],
        "final_reason": state["final_reason"]
    })
    state["moderation_history"] = history
    
    state["status"] = "finalized"
    
    print(f"   Decision: {state['final_decision']}")
    print(f"   Reason: {state['final_reason']}")
    
    return state

# ============================================
# STEP 3: ROUTING FUNCTIONS
# ============================================

def route_after_auto(state: ModerationState) -> Literal["approve", "reject", "human"]:
    """Route after auto-moderation."""
    
    decision = state.get("auto_decision", "needs_review")
    
    if decision == "approved":
        return "approve"
    elif decision == "rejected":
        return "reject"
    else:
        return "human"

# ============================================
# STEP 4: BUILD THE GRAPH (FIXED)
# ============================================

def build_moderation_graph(checkpointer=None):
    """
    Build the content moderation graph.
    
    Args:
        checkpointer: Optional checkpointer for persistence
    """
    graph = StateGraph(ModerationState)
    
    # Add nodes
    graph.add_node("auto_moderate", auto_moderate)
    graph.add_node("human_review", human_review)
    graph.add_node("final_decision", make_final_decision)
    
    # Set entry point
    graph.set_entry_point("auto_moderate")
    
    # Conditional routing from auto_moderate
    graph.add_conditional_edges(
        "auto_moderate",
        route_after_auto,
        {
            "approve": "final_decision",
            "reject": "final_decision",
            "human": "human_review"
        }
    )
    
    # Human review goes to final decision
    graph.add_edge("human_review", "final_decision")
    
    # Final decision ends
    graph.add_edge("final_decision", END)
    
    # Compile with checkpointer if provided
    return graph.compile(checkpointer=checkpointer)

# ============================================
# STEP 5: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("🔍 CONTENT MODERATION GRAPH")
    print("=" * 60)
    
    # Create checkpointer for persistence
    checkpointer = MemorySaver()
    
    # Build graph WITH checkpointer
    graph_with_checkpoint = build_moderation_graph(checkpointer=checkpointer)
    
    # Test posts
    test_posts = [
        {
            "id": "1",
            "content": "I love this community! It's so helpful.",
            "author": "user1"
        },
        {
            "id": "2",
            "content": "This is spam spam spam! Click here for free money.",
            "author": "user2"
        },
        {
            "id": "3",
            "content": "This is borderline content that might need review.",
            "author": "user3"
        }
    ]
    
    for post in test_posts:
        thread_id = f"moderation_{post['id']}"
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "post_id": post["id"],
            "post_content": post["content"],
            "author_id": post["author"],
            "auto_score": 0.0,
            "auto_reason": "",
            "auto_decision": "",
            "human_decision": None,
            "human_reviewer": None,
            "human_comment": None,
            "final_decision": "",
            "final_reason": "",
            "status": "pending",
            "moderation_history": []
        }
        
        print("\n" + "=" * 60)
        print(f"📝 Post {post['id']}: {post['content'][:50]}...")
        print("=" * 60)
        
        final_state = graph_with_checkpoint.invoke(initial_state, config)
        
        print("\n📊 Final Decision:")
        print(f"   Decision: {final_state.get('final_decision', 'unknown')}")
        print(f"   Reason: {final_state.get('final_reason', 'No reason')}")
        
        # Show history
        history = final_state.get("moderation_history", [])
        if history:
            print("\n📋 Moderation History:")
            for entry in history:
                print(f"   • {entry['final_decision']}: {entry['final_reason']}")

if __name__ == "__main__":
    run_demo()
