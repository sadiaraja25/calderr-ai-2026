#!/usr/bin/env python3
"""
Week 4 - Monday: Lab 4.1 - Document Processing Graph
With conditional edge: if doc too large, split into parts first.
"""

import os
import uuid
from typing import TypedDict, List, Dict, Any, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class DocumentState(TypedDict):
    """State for document processing graph."""
    
    # Input
    document_id: str
    content: str
    
    # Validation
    is_valid: bool
    validation_message: str
    doc_size: int
    
    # Processing
    chunks: List[str]
    chunk_count: int
    embeddings: List[List[float]]
    
    # Status
    status: str
    errors: List[str]

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def load_document(state: DocumentState) -> DocumentState:
    """Load document and initialize state."""
    print("📄 [LOAD] Loading document...")
    
    content = state.get("content", "")
    
    state["document_id"] = str(uuid.uuid4())[:8]
    state["doc_size"] = len(content)
    state["chunks"] = []
    state["chunk_count"] = 0
    state["status"] = "loading"
    state["errors"] = []
    
    print(f"   Doc ID: {state['document_id']}")
    print(f"   Size: {state['doc_size']} characters")
    
    return state

def validate_document(state: DocumentState) -> DocumentState:
    """Validate the document (size, content)."""
    print("🔍 [VALIDATE] Validating document...")
    
    content = state.get("content", "")
    errors = []
    
    # Check if empty
    if not content.strip():
        errors.append("Document is empty")
        state["is_valid"] = False
        state["validation_message"] = "Empty document"
        state["status"] = "rejected"
        return state
    
    # Check minimum size
    if len(content) < 10:
        errors.append("Document too short (minimum 10 characters)")
        state["is_valid"] = False
        state["validation_message"] = "Too short"
        state["status"] = "rejected"
        return state
    
    # Check maximum size (for this lab, 5000 chars is "large")
    if len(content) > 5000:
        print("   ⚠️ Document is LARGE (>5000 chars)")
        state["validation_message"] = "large"
    else:
        print("   ✅ Document is valid")
        state["validation_message"] = "valid"
    
    state["is_valid"] = True
    state["status"] = "validated"
    state["errors"] = errors
    
    return state

def chunk_document(state: DocumentState) -> DocumentState:
    """Chunk the document into pieces."""
    print("✂️ [CHUNK] Chunking document...")
    
    content = state.get("content", "")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    
    state["chunks"] = chunks
    state["chunk_count"] = len(chunks)
    state["status"] = "chunked"
    
    print(f"   Created {len(chunks)} chunks")
    return state

def split_large_document(state: DocumentState) -> DocumentState:
    """Special splitter for large documents."""
    print("🔀 [SPLIT] Splitting large document into parts...")
    
    content = state.get("content", "")
    
    # Split into parts based on size
    part_size = 2000
    parts = []
    for i in range(0, len(content), part_size):
        parts.append(content[i:i+part_size])
    
    # Now chunk each part
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    for part in parts:
        chunks = text_splitter.split_text(part)
        all_chunks.extend(chunks)
    
    state["chunks"] = all_chunks
    state["chunk_count"] = len(all_chunks)
    state["status"] = "split_and_chunked"
    
    print(f"   Split into {len(parts)} parts, {len(all_chunks)} total chunks")
    return state

def embed_chunks(state: DocumentState) -> DocumentState:
    """Embed all chunks (simulated)."""
    print("📊 [EMBED] Embedding chunks...")
    
    chunks = state.get("chunks", [])
    
    # In a real scenario, you'd use sentence-transformers here
    # For this demo, we'll simulate embeddings
    embeddings = []
    for i, chunk in enumerate(chunks):
        # Simulate embedding (mock)
        fake_embedding = [float(i) / 100.0 for _ in range(8)]
        embeddings.append(fake_embedding)
    
    state["embeddings"] = embeddings
    state["status"] = "embedded"
    
    print(f"   Created {len(embeddings)} embeddings")
    return state

def confirm_completion(state: DocumentState) -> DocumentState:
    """Confirm processing is complete."""
    print("✅ [CONFIRM] Processing complete!")
    
    state["status"] = "complete"
    
    print(f"   Document ID: {state['document_id']}")
    print(f"   Chunks: {state['chunk_count']}")
    print(f"   Embeddings: {len(state.get('embeddings', []))}")
    
    return state

# ============================================
# STEP 3: CONDITIONAL ROUTING FUNCTIONS
# ============================================

def route_after_validation(state: DocumentState) -> Literal["chunk", "split", "reject"]:
    """Decide which path to take based on document validation."""
    
    if not state.get("is_valid", False):
        print("   ❌ Document invalid → Rejecting")
        return "reject"
    
    validation_msg = state.get("validation_message", "valid")
    
    if validation_msg == "large":
        print("   📏 Document is large → Splitting")
        return "split"
    
    print("   ✅ Document is valid → Chunking")
    return "chunk"

# ============================================
# STEP 4: BUILD THE GRAPH
# ============================================

def build_document_graph():
    """Build the complete document processing graph."""
    
    # Create graph
    graph = StateGraph(DocumentState)
    
    # Add nodes
    graph.add_node("load", load_document)
    graph.add_node("validate", validate_document)
    graph.add_node("chunk", chunk_document)
    graph.add_node("split", split_large_document)
    graph.add_node("embed", embed_chunks)
    graph.add_node("confirm", confirm_completion)
    
    # Add a reject node (simple node that sets status to rejected)
    def reject_document(state: DocumentState) -> DocumentState:
        state["status"] = "rejected"
        print("   ❌ Document rejected")
        return state
    
    graph.add_node("reject", reject_document)
    
    # Set entry point
    graph.set_entry_point("load")
    
    # Add edges
    graph.add_edge("load", "validate")
    
    # Conditional edge: validate → chunk OR split OR reject
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "chunk": "chunk",
            "split": "split",
            "reject": "reject"
        }
    )
    
    # After chunk or split, go to embed
    graph.add_edge("chunk", "embed")
    graph.add_edge("split", "embed")
    
    # After embed, confirm
    graph.add_edge("embed", "confirm")
    
    # After confirm or reject, end
    graph.add_edge("confirm", END)
    graph.add_edge("reject", END)
    
    return graph.compile()

# ============================================
# STEP 5: RUN THE GRAPH
# ============================================

def run_demo():
    print("=" * 60)
    print("📊 LAB 4.1: DOCUMENT PROCESSING GRAPH")
    print("=" * 60)
    
    # Test cases
    test_documents = {
        "small": "This is a small document. It has just enough content to be valid.",
        "large": """
        This is a large document that exceeds the size limit. 
        """ * 300 + "End of large document.",
        "empty": "",
        "short": "Hi"
    }
    
    for name, content in test_documents.items():
        print("\n" + "=" * 60)
        print(f"📄 Test Case: {name.upper()}")
        print("=" * 60)
        print(f"Content length: {len(content)} characters")
        
        initial_state = {
            "document_id": "",
            "content": content,
            "is_valid": True,
            "validation_message": "",
            "doc_size": 0,
            "chunks": [],
            "chunk_count": 0,
            "embeddings": [],
            "status": "pending",
            "errors": []
        }
        
        graph = build_document_graph()
        
        try:
            final_state = graph.invoke(initial_state)
            print("\n📊 Final Status:", final_state["status"])
            print(f"   Chunks: {final_state.get('chunk_count', 0)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_demo()
