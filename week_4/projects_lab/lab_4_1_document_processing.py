#!/usr/bin/env python3
"""
Lab 4.1: Document Processing Graph
Concept: Monday - LangGraph Foundations
State → Node → Edge with Conditional Routing
"""

import os
import uuid
from typing import TypedDict, Literal, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================
# STEP 1: DEFINE STATE
# ============================================

class DocumentState(TypedDict):
    """State flows through the graph."""
    
    # Input
    content: str
    
    # Processing
    chunks: List[str]
    chunk_count: int
    embeddings: List[List[float]]
    
    # Status
    doc_id: str
    status: str
    doc_size: int
    is_valid: bool

# ============================================
# STEP 2: NODES (Functions)
# ============================================

def load_document(state: DocumentState) -> DocumentState:
    """Node 1: Load the document."""
    print("📄 [LOAD] Loading document...")
    
    content = state.get("content", "")
    
    state["doc_id"] = str(uuid.uuid4())[:8]
    state["doc_size"] = len(content)
    state["status"] = "loaded"
    state["is_valid"] = True
    
    print(f"   Doc ID: {state['doc_id']}")
    print(f"   Size: {state['doc_size']} characters")
    return state

def validate_document(state: DocumentState) -> DocumentState:
    """Node 2: Validate the document."""
    print("🔍 [VALIDATE] Validating document...")
    
    content = state.get("content", "")
    
    if not content.strip():
        state["is_valid"] = False
        state["status"] = "invalid"
        print("   ❌ Document is empty")
        return state
    
    if len(content) < 10:
        state["is_valid"] = False
        state["status"] = "invalid"
        print("   ❌ Document too short")
        return state
    
    state["status"] = "validated"
    print("   ✅ Document is valid")
    return state

def chunk_document(state: DocumentState) -> DocumentState:
    """Node 3a: Chunk the document."""
    print("✂️ [CHUNK] Chunking document...")
    
    content = state.get("content", "")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_text(content)
    
    state["chunks"] = chunks
    state["chunk_count"] = len(chunks)
    state["status"] = "chunked"
    
    print(f"   Created {len(chunks)} chunks")
    return state

def split_large_document(state: DocumentState) -> DocumentState:
    """Node 3b: Split large document before chunking."""
    print("🔀 [SPLIT] Splitting large document...")
    
    content = state.get("content", "")
    
    # Split into parts
    parts = []
    part_size = 2000
    for i in range(0, len(content), part_size):
        parts.append(content[i:i+part_size])
    
    # Chunk each part
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    for part in parts:
        chunks = splitter.split_text(part)
        all_chunks.extend(chunks)
    
    state["chunks"] = all_chunks
    state["chunk_count"] = len(all_chunks)
    state["status"] = "split_and_chunked"
    
    print(f"   Split into {len(parts)} parts, {len(all_chunks)} chunks")
    return state

def embed_chunks(state: DocumentState) -> DocumentState:
    """Node 4: Embed chunks."""
    print("📊 [EMBED] Embedding chunks...")
    
    chunks = state.get("chunks", [])
    
    # Simulate embeddings (in real use, call sentence-transformers)
    embeddings = [[float(i) / 100.0 for _ in range(8)] for i in range(len(chunks))]
    
    state["embeddings"] = embeddings
    state["status"] = "embedded"
    
    print(f"   Created {len(embeddings)} embeddings")
    return state

def confirm_completion(state: DocumentState) -> DocumentState:
    """Node 5: Confirm completion."""
    print("✅ [CONFIRM] Processing complete!")
    
    state["status"] = "complete"
    print(f"   Document ID: {state['doc_id']}")
    print(f"   Chunks: {state['chunk_count']}")
    print(f"   Embeddings: {len(state.get('embeddings', []))}")
    
    return state

def reject_document(state: DocumentState) -> DocumentState:
    """Reject invalid document."""
    print("🚫 [REJECT] Document rejected.")
    state["status"] = "rejected"
    return state

# ============================================
# STEP 3: ROUTING FUNCTION
# ============================================

def route_after_validate(state: DocumentState) -> Literal["chunk", "split", "reject"]:
    """Conditional edge: decide path based on document."""
    
    if not state.get("is_valid", False):
        return "reject"
    
    # If doc > 5000 chars, split it
    if state.get("doc_size", 0) > 5000:
        print("   📏 Large document → Splitting")
        return "split"
    
    print("   📄 Normal document → Chunking")
    return "chunk"

# ============================================
# STEP 4: BUILD GRAPH
# ============================================

def build_graph():
    """Build the document processing graph."""
    
    graph = StateGraph(DocumentState)
    
    # Add nodes
    graph.add_node("load", load_document)
    graph.add_node("validate", validate_document)
    graph.add_node("chunk", chunk_document)
    graph.add_node("split", split_large_document)
    graph.add_node("embed", embed_chunks)
    graph.add_node("confirm", confirm_completion)
    graph.add_node("reject", reject_document)
    
    # Set entry point
    graph.set_entry_point("load")
    
    # Edges
    graph.add_edge("load", "validate")
    
    # Conditional edge
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "chunk": "chunk",
            "split": "split",
            "reject": "reject"
        }
    )
    
    # Continue paths
    graph.add_edge("chunk", "embed")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "confirm")
    graph.add_edge("confirm", END)
    graph.add_edge("reject", END)
    
    return graph.compile()

# ============================================
# STEP 5: TEST
# ============================================

def test_graph():
    print("=" * 60)
    print("🧪 Lab 4.1: Document Processing Graph")
    print("=" * 60)
    
    graph = build_graph()
    
    test_docs = [
        ("Small Doc", "This is a small document."),
        ("Large Doc", "This is a large document. " * 300),
        ("Empty Doc", ""),
    ]
    
    for name, content in test_docs:
        print(f"\n📝 Testing: {name} ({len(content)} chars)")
        print("-" * 40)
        
        state = {
            "content": content,
            "chunks": [],
            "chunk_count": 0,
            "embeddings": [],
            "doc_id": "",
            "status": "pending",
            "doc_size": 0,
            "is_valid": True
        }
        
        result = graph.invoke(state)
        print(f"✅ Status: {result['status']}")
        print(f"   Chunks: {result.get('chunk_count', 0)}")

if __name__ == "__main__":
    test_graph()
