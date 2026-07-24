#!/usr/bin/env python3
"""
Week 4 - Wednesday: Research Workflow Graph
State: query, search_results, draft, feedback, final_report
Iterates on draft until quality threshold is met.
"""

import os
import json
from typing import TypedDict, List, Annotated, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class ResearchState(TypedDict):
    """State for the research workflow."""
    
    # Input
    query: str
    
    # Search results
    search_results: List[str]
    
    # Outline
    outline: str
    
    # Draft and iterations
    current_draft: str
    draft_history: List[str]  # Accumulates drafts
    revision_feedback: List[str]
    
    # Quality
    quality_score: float  # 0-1
    passes_quality: bool
    
    # Final
    final_report: str
    
    # Status
    status: str
    iteration_count: int

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def research_topic(state: ResearchState) -> ResearchState:
    """Research the topic and gather information."""
    print("🔍 [RESEARCH] Researching topic...")
    
    query = state.get("query", "")
    
    # Simulate search results
    search_results = [
        f"Search result 1 for '{query}': Key information about the topic.",
        f"Search result 2 for '{query}': Additional context and details.",
        f"Search result 3 for '{query}': Recent developments and applications.",
        f"Search result 4 for '{query}': Related concepts and frameworks.",
        f"Search result 5 for '{query}': Common challenges and solutions."
    ]
    
    state["search_results"] = search_results
    state["status"] = "researched"
    
    print(f"   Found {len(search_results)} results")
    return state

def create_outline(state: ResearchState) -> ResearchState:
    """Create an outline based on research."""
    print("📋 [OUTLINE] Creating outline...")
    
    query = state.get("query", "")
    results = state.get("search_results", [])
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        Create a detailed outline for a research article about:
        
        Topic: {query}
        
        Research findings: {json.dumps(results[:3], indent=2)}
        
        Outline format:
        I. Introduction
        II. Key Concepts
        III. Applications
        IV. Challenges
        V. Conclusion
        
        Outline:"""
        
        response = llm.invoke(prompt)
        
        state["outline"] = response.content
        state["status"] = "outlined"
        
        print(f"   Outline length: {len(response.content)} characters")
        print(f"   Outline preview: {response.content[:100]}...")
        
    except Exception as e:
        state["outline"] = f"Error creating outline: {e}"
        state["status"] = "outline_error"
    
    return state

def write_draft(state: ResearchState) -> ResearchState:
    """Write a draft based on the outline."""
    print("✍️ [DRAFT] Writing draft...")
    
    query = state.get("query", "")
    outline = state.get("outline", "")
    results = state.get("search_results", [])
    
    # Get previous draft if this is a revision
    previous_draft = state.get("current_draft", "")
    feedback = state.get("revision_feedback", [])
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.5,
            api_key=GROQ_API_KEY
        )
        
        if previous_draft and feedback:
            prompt = f"""
            Revise this draft based on feedback.
            
            Topic: {query}
            
            Previous Draft:
            {previous_draft}
            
            Feedback: {json.dumps(feedback, indent=2)}
            
            Please create an improved version addressing the feedback.
            """
        else:
            prompt = f"""
            Write a research article based on this outline.
            
            Topic: {query}
            
            Outline:
            {outline}
            
            Research Results:
            {json.dumps(results[:3], indent=2)}
            
            Article:"""
        
        response = llm.invoke(prompt)
        
        # Save current draft to history
        draft_history = state.get("draft_history", [])
        if previous_draft:
            draft_history.append(previous_draft)
            state["draft_history"] = draft_history
        
        state["current_draft"] = response.content
        state["status"] = "drafted"
        
        print(f"   Draft length: {len(response.content)} characters")
        print(f"   Draft preview: {response.content[:100]}...")
        
    except Exception as e:
        state["current_draft"] = f"Error writing draft: {e}"
        state["status"] = "draft_error"
    
    return state

def evaluate_quality(state: ResearchState) -> ResearchState:
    """Evaluate the draft quality."""
    print("📊 [EVALUATE] Evaluating draft quality...")
    
    draft = state.get("current_draft", "")
    query = state.get("query", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        Evaluate this research draft on a scale of 0-1.
        
        Query: {query}
        
        Draft:
        {draft[:500]}...
        
        Rate on:
        1. Completeness (0-1): Does it cover the topic fully?
        2. Clarity (0-1): Is it well-written and clear?
        3. Accuracy (0-1): Does it contain correct information?
        
        Return ONLY JSON:
        {{"completeness": 0.8, "clarity": 0.7, "accuracy": 0.9}}
        """
        
        response = llm.invoke(prompt)
        
        # Extract scores from response
        import re
        import json
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            avg_score = sum(scores.values()) / len(scores)
            state["quality_score"] = avg_score
            
            # Check if quality threshold met (0.7)
            state["passes_quality"] = avg_score >= 0.7
            state["status"] = "evaluated"
            
            print(f"   Quality score: {avg_score:.2f}")
            print(f"   Passes threshold (0.7): {state['passes_quality']}")
        else:
            state["quality_score"] = 0.5
            state["passes_quality"] = False
            state["status"] = "evaluation_error"
        
    except Exception as e:
        state["quality_score"] = 0.5
        state["passes_quality"] = False
        state["status"] = "evaluation_error"
        print(f"   ❌ Evaluation error: {e}")
    
    return state

def revise_draft(state: ResearchState) -> ResearchState:
    """Provide feedback for revision."""
    print("🔄 [REVISE] Preparing revision feedback...")
    
    quality_score = state.get("quality_score", 0.5)
    draft = state.get("current_draft", "")
    
    feedback = []
    
    if quality_score < 0.7:
        if quality_score < 0.5:
            feedback.append("Draft needs significant improvement")
        else:
            feedback.append("Draft needs minor improvements")
    
    # Add specific feedback based on quality
    if "completeness" in str(state):
        feedback.append("Add more depth to key sections")
    
    if "clarity" in str(state):
        feedback.append("Improve sentence structure and flow")
    
    if "accuracy" in str(state):
        feedback.append("Verify and strengthen factual claims")
    
    # Store feedback
    revision_feedback = state.get("revision_feedback", [])
    revision_feedback.extend(feedback)
    state["revision_feedback"] = revision_feedback
    state["status"] = "revised"
    
    print(f"   Feedback provided: {len(feedback)} items")
    for f in feedback:
        print(f"     - {f}")
    
    return state

def finalize_report(state: ResearchState) -> ResearchState:
    """Finalize the research report."""
    print("✅ [FINALIZE] Creating final report...")
    
    draft = state.get("current_draft", "")
    outline = state.get("outline", "")
    feedback = state.get("revision_feedback", [])
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        Create a polished final research report.
        
        Draft: {draft[:1000]}...
        
        Feedback addressed: {json.dumps(feedback, indent=2)}
        
        Final Report:"""
        
        response = llm.invoke(prompt)
        
        state["final_report"] = response.content
        state["status"] = "complete"
        
        print(f"   Final report length: {len(response.content)} characters")
        print(f"   Final report preview: {response.content[:100]}...")
        
    except Exception as e:
        state["final_report"] = f"Error finalizing report: {e}"
        state["status"] = "final_error"
    
    return state

# ============================================
# STEP 3: ROUTING FUNCTION
# ============================================

def should_continue(state: ResearchState) -> Literal["research", "outline", "draft", "evaluate", "revise", "final"]:
    """Decide what to do next based on state."""
    
    status = state.get("status", "")
    passes = state.get("passes_quality", False)
    iteration = state.get("iteration_count", 0)
    
    # If complete, stop
    if status == "complete":
        return "final"
    
    # If status is pending, start research
    if status == "" or status == "pending":
        return "research"
    
    # If researched, create outline
    if status == "researched":
        return "outline"
    
    # If outlined, write draft
    if status == "outlined":
        return "draft"
    
    # If drafted, evaluate
    if status == "drafted":
        return "evaluate"
    
    # If evaluated
    if status == "evaluated":
        # If passes quality, finalize
        if passes:
            return "final"
        else:
            # If max iterations reached, finalize anyway
            if iteration >= 3:
                return "final"
            # Otherwise revise and re-draft
            return "revise"
    
    # If revised, go back to draft
    if status == "revised":
        return "draft"
    
    return "final"

# ============================================
# STEP 4: BUILD THE GRAPH
# ============================================

def build_research_graph():
    """Build the research workflow graph."""
    
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("research", research_topic)
    graph.add_node("outline", create_outline)
    graph.add_node("draft", write_draft)
    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("revise", revise_draft)
    graph.add_node("final", finalize_report)
    
    # Set entry point
    graph.set_entry_point("research")
    
    # Add edges
    graph.add_edge("research", "outline")
    graph.add_edge("outline", "draft")
    graph.add_edge("draft", "evaluate")
    
    # Conditional routing from evaluate
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "revise": "revise",
            "final": "final"
        }
    )
    
    # Revise goes back to draft
    graph.add_edge("revise", "draft")
    
    # Final ends
    graph.add_edge("final", END)
    
    return graph.compile()

# ============================================
# STEP 5: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("📝 RESEARCH WORKFLOW GRAPH")
    print("=" * 60)
    
    test_queries = [
        "The impact of RAG on AI systems",
        "LangGraph for agent orchestration"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"📝 Query: {query}")
        print("-" * 40)
        
        initial_state = {
            "query": query,
            "search_results": [],
            "outline": "",
            "current_draft": "",
            "draft_history": [],
            "revision_feedback": [],
            "quality_score": 0.0,
            "passes_quality": False,
            "final_report": "",
            "status": "pending",
            "iteration_count": 0
        }
        
        graph = build_research_graph()
        final_state = graph.invoke(initial_state)
        
        print("\n📊 Final Results:")
        print(f"   Status: {final_state.get('status', 'unknown')}")
        print(f"   Quality Score: {final_state.get('quality_score', 0):.2f}")
        print(f"   Iterations: {final_state.get('iteration_count', 0)}")
        print(f"   Drafts Created: {len(final_state.get('draft_history', [])) + 1}")
        print(f"   Feedback Items: {len(final_state.get('revision_feedback', []))}")
        
        print(f"\n📄 Final Report:\n{final_state.get('final_report', 'No report')[:300]}...")

if __name__ == "__main__":
    run_demo()
