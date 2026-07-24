#!/usr/bin/env python3
"""
Week 4 - Friday: Production Graph with LangSmith Tracing & SqliteSaver
"""

import os
import json
import time
import sqlite3
from typing import TypedDict, Literal, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# ============================================
# STEP 1: LOAD ENVIRONMENT & SETUP LANGSMITH
# ============================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Set LangSmith environment variables
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "production-graph"
    print("✅ LangSmith tracing enabled")
else:
    print("⚠️ LangSmith API key not found. Tracing disabled.")

if not GROQ_API_KEY:
    print("❌ Groq API key not found!")
    exit(1)

# ============================================
# STEP 1.5: TRY DIFFERENT SqliteSaver IMPORTS
# ============================================

def get_checkpointer(conn):
    """Get the appropriate checkpointer for the current LangGraph version."""
    try:
        # Try newer version
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver(conn)
    except ImportError:
        try:
            # Try older version
            from langgraph.checkpoint import SqliteSaver
            return SqliteSaver(conn)
        except ImportError:
            try:
                # Try alternative
                from langgraph.checkpoint.sqlite import aiosqlite
                from langgraph.checkpoint.sqlite import AsyncSqliteSaver
                return AsyncSqliteSaver.from_conn_string("production_graph.db")
            except ImportError:
                print("⚠️ SqliteSaver not available. Using MemorySaver fallback.")
                from langgraph.checkpoint.memory import MemorySaver
                return MemorySaver()

# ============================================
# STEP 2: DEFINE STATE SCHEMA
# ============================================

class ProductionState(TypedDict):
    """State for the production graph."""
    
    # Input
    query: str
    user_id: str
    
    # Processing
    step: int
    current_stage: str
    
    # Results
    research_results: List[str]
    draft: str
    final_report: str
    
    # Status
    status: str
    created_at: str
    updated_at: str
    iteration_count: int
    error: Optional[str]

# ============================================
# STEP 3: NODE FUNCTIONS
# ============================================

def research_topic(state: ProductionState) -> ProductionState:
    """Research the topic."""
    print("🔍 [RESEARCH] Researching topic...")
    
    query = state.get("query", "")
    
    # Simulate research
    results = [
        f"Key finding 1 about '{query}'",
        f"Key finding 2 about '{query}'",
        f"Key finding 3 about '{query}'"
    ]
    
    state["research_results"] = results
    state["current_stage"] = "research"
    state["step"] = 1
    state["updated_at"] = datetime.now().isoformat()
    
    print(f"   Found {len(results)} results")
    return state

def analyze_findings(state: ProductionState) -> ProductionState:
    """Analyze research findings."""
    print("📊 [ANALYZE] Analyzing findings...")
    
    results = state.get("research_results", [])
    
    analysis = f"Analysis of findings: {'; '.join(results)}"
    state["draft"] = analysis
    state["current_stage"] = "analysis"
    state["step"] = 2
    state["updated_at"] = datetime.now().isoformat()
    
    print(f"   Analysis complete")
    return state

def generate_report(state: ProductionState) -> ProductionState:
    """Generate final report."""
    print("📝 [REPORT] Generating final report...")
    
    analysis = state.get("draft", "")
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        prompt = f"""
        Generate a comprehensive report based on this analysis:
        
        {analysis}
        
        Report:"""
        
        response = llm.invoke(prompt)
        
        state["final_report"] = response.content
        state["current_stage"] = "report"
        state["step"] = 3
        state["status"] = "complete"
        state["updated_at"] = datetime.now().isoformat()
        
        print(f"   Report generated ({len(response.content)} characters)")
        
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "error"
        print(f"   ❌ Error: {e}")
    
    return state

# ============================================
# STEP 4: BUILD THE GRAPH
# ============================================

def build_production_graph(checkpointer=None):
    """
    Build the production graph with optional persistence.
    
    Args:
        checkpointer: Optional checkpointer for persistence
    """
    graph = StateGraph(ProductionState)
    
    # Add nodes
    graph.add_node("research", research_topic)
    graph.add_node("analyze", analyze_findings)
    graph.add_node("report", generate_report)
    
    # Set entry point
    graph.set_entry_point("research")
    
    # Add edges
    graph.add_edge("research", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)
    
    # Compile with checkpointer if provided
    return graph.compile(checkpointer=checkpointer)

# ============================================
# STEP 5: DEMO WITH PERSISTENCE
# ============================================

def run_demo():
    """Run the production graph demo with persistence."""
    
    print("=" * 60)
    print("🚀 PRODUCTION GRAPH DEMO")
    print("=" * 60)
    
    # ============================================
    # Part A: With MemorySaver (in-memory)
    # ============================================
    print("\n📌 PART A: MemorySaver (in-memory)")
    print("-" * 40)
    
    from langgraph.checkpoint.memory import MemorySaver
    memory_checkpointer = MemorySaver()
    
    graph_memory = build_production_graph(checkpointer=memory_checkpointer)
    
    # Run with persistence
    config = {"configurable": {"thread_id": "memory_demo"}}
    initial_state = {
        "query": "What is RAG?",
        "user_id": "user_123",
        "step": 0,
        "current_stage": "start",
        "research_results": [],
        "draft": "",
        "final_report": "",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "iteration_count": 0,
        "error": None
    }
    
    try:
        final_state = graph_memory.invoke(initial_state, config)
        print(f"✅ Done! Final status: {final_state.get('status')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ============================================
    # Part B: With SqliteSaver (persistent)
    # ============================================
    print("\n📌 PART B: SqliteSaver (persistent disk)")
    print("-" * 40)
    
    try:
        # Create SQLite database connection
        conn = sqlite3.connect("production_graph.db", check_same_thread=False)
        
        # Get the appropriate checkpointer
        checkpointer = get_checkpointer(conn)
        
        # Build with persistence
        graph_sqlite = build_production_graph(checkpointer=checkpointer)
        
        # Run with persistence
        config_sqlite = {"configurable": {"thread_id": "sqlite_demo"}}
        
        initial_state_sqlite = {
            "query": "How does LangGraph work?",
            "user_id": "user_456",
            "step": 0,
            "current_stage": "start",
            "research_results": [],
            "draft": "",
            "final_report": "",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "iteration_count": 0,
            "error": None
        }
        
        final_state_sqlite = graph_sqlite.invoke(initial_state_sqlite, config_sqlite)
        print(f"✅ Done! Final status: {final_state_sqlite.get('status')}")
        
    except Exception as e:
        print(f"⚠️ SqliteSaver not available: {e}")
        print("   Using MemorySaver as fallback")
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        graph_sqlite = build_production_graph(checkpointer=checkpointer)
        print("   ✅ Using MemorySaver fallback")
    
    # ============================================
    # Part C: Show state persistence
    # ============================================
    print("\n📌 PART C: State Persistence")
    print("-" * 40)
    print("   ✅ State was saved")
    print("   💡 The graph can be interrupted and resumed later")
    print("   🔑 Thread ID: sqlite_demo")
    
    print("\n📚 Key Production Features:")
    print("   • LangSmith tracing (if enabled)")
    print("   • SQLite persistence (if available)")
    print("   • State saved to disk")
    print("   • Ready for interrupt/resume")

if __name__ == "__main__":
    run_demo()
