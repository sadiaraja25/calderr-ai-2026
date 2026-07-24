#!/usr/bin/env python3
"""
Week 4 - Monday: First LangGraph - Linear 3-Node Pipeline
"""

import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class DocumentState(TypedDict):
    """State that flows through the graph."""
    
    # Input
    content: str
    
    # Processed data
    chunks: List[str]
    chunk_count: int
    
    # Result
    summary: str
    status: str

# ============================================
# STEP 2: DEFINE NODES (Functions)
# ============================================

def load_document(state: DocumentState) -> DocumentState:
    """Node 1: Load the document."""
    print("📄 Loading document...")
    
    # In a real scenario, you'd load from a file
    # Here we just pass through the content
    content = state.get("content", "")
    
    # Update state
    state["content"] = content
    state["status"] = "loaded"
    
    print(f"   Content length: {len(content)} characters")
    return state

def chunk_document(state: DocumentState) -> DocumentState:
    """Node 2: Chunk the document."""
    print("✂️ Chunking document...")
    
    content = state.get("content", "")
    
    # Simple chunking (in real code, use RecursiveCharacterTextSplitter)
    chunks = []
    chunk_size = 100
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i+chunk_size])
    
    state["chunks"] = chunks
    state["chunk_count"] = len(chunks)
    
    print(f"   Created {len(chunks)} chunks")
    return state

def summarize_document(state: DocumentState) -> DocumentState:
    """Node 3: Summarize the document (using Groq)."""
    print("🤖 Summarizing document...")
    
    chunks = state.get("chunks", [])
    
    if not chunks:
        state["summary"] = "No content to summarize"
        state["status"] = "failed"
        return state
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        # Combine first few chunks for summary
        text = " ".join(chunks[:3])
        
        prompt = f"Summarize this text in 2-3 sentences:\n\n{text}"
        response = llm.invoke(prompt)
        
        state["summary"] = response.content
        state["status"] = "complete"
        
        print(f"   Summary: {state['summary'][:100]}...")
        
    except Exception as e:
        state["summary"] = f"Error: {e}"
        state["status"] = "error"
    
    return state

# ============================================
# STEP 3: BUILD THE GRAPH
# ============================================

def build_graph():
    """Build the document processing graph."""
    
    # 3a. Create the graph with the state schema
    graph = StateGraph(DocumentState)
    
    # 3b. Add nodes
    graph.add_node("load", load_document)
    graph.add_node("chunk", chunk_document)
    graph.add_node("summarize", summarize_document)
    
    # 3c. Add edges (order matters)
    graph.set_entry_point("load")
    graph.add_edge("load", "chunk")
    graph.add_edge("chunk", "summarize")
    graph.add_edge("summarize", END)
    
    # 3d. Compile the graph
    return graph.compile()

# ============================================
# STEP 4: RUN THE GRAPH
# ============================================

def run_demo():
    """Run the graph with sample data."""
    
    print("=" * 60)
    print("📊 FIRST LANGGRAPH DEMO")
    print("=" * 60)
    
    # Sample content
    sample_content = """
    LangGraph is a library for building stateful, multi-actor applications 
    with LLMs. It extends LangChain to allow for cyclic, multi-step workflows 
    with branching, looping, and human-in-the-loop capabilities. 
    LangGraph uses a graph-based architecture where nodes are functions 
    that process state, and edges define how state flows between nodes.
    """
    
    # Initial state
    initial_state = {
        "content": sample_content,
        "chunks": [],
        "chunk_count": 0,
        "summary": "",
        "status": "pending"
    }
    
    # Build and run the graph
    graph = build_graph()
    
    # Visualize the graph structure
    print("\n🔗 Graph Structure:")
    print("   load → chunk → summarize → END")
    
    print("\n⚡ Running graph...")
    print("-" * 40)
    
    # Execute
    final_state = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("📊 FINAL STATE")
    print("=" * 60)
    print(f"Content length: {len(final_state['content'])} characters")
    print(f"Chunks created: {final_state['chunk_count']}")
    print(f"Status: {final_state['status']}")
    print(f"\nSummary:\n{final_state['summary']}")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
