#!/usr/bin/env python3
"""
Week 4 - Wednesday: Multi-Turn Stateful Agent
Accumulates messages, tool calls, and intermediate results.
"""

import os
from typing import TypedDict, List, Annotated, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================
# STEP 1: DEFINE STATE SCHEMA
# ============================================

class AgentState(TypedDict):
    """State for the multi-turn agent."""
    
    # Messages (accumulates - uses add_messages reducer)
    messages: Annotated[List, add_messages]
    
    # Query (overwrites)
    query: str
    
    # Accumulating fields (we'll handle manually)
    tool_calls: List[dict]
    intermediate_results: List[str]
    
    # Status
    current_response: str
    status: str
    iteration_count: int

# ============================================
# STEP 2: NODE FUNCTIONS
# ============================================

def process_query(state: AgentState) -> AgentState:
    """Process the user query."""
    print("🔍 [PROCESS] Processing query...")
    
    query = state.get("query", "")
    
    # Add user message to state
    state["messages"] = state.get("messages", []) + [HumanMessage(content=query)]
    state["status"] = "processing"
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    print(f"   Query: {query[:50]}...")
    print(f"   Iteration: {state['iteration_count']}")
    
    return state

def call_llm(state: AgentState) -> AgentState:
    """Call the LLM with conversation history."""
    print("🤖 [LLM] Calling LLM...")
    
    # Add system message if not already there
    messages = state.get("messages", [])
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(content="""
        You are a helpful assistant. You have access to tools.
        Respond to user queries and track what you've done.
        """)
        messages = [system_msg] + messages
    
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=GROQ_API_KEY
        )
        
        # Call LLM with all messages
        response = llm.invoke(messages)
        
        # Add AI response to messages
        state["messages"] = messages + [response]
        state["current_response"] = response.content
        state["status"] = "llm_complete"
        
        print(f"   Response: {response.content[:100]}...")
        print(f"   Total messages: {len(state['messages'])}")
        
    except Exception as e:
        state["current_response"] = f"Error: {e}"
        state["status"] = "error"
    
    return state

def simulate_tool_call(state: AgentState) -> AgentState:
    """Simulate a tool call and add result."""
    print("🔧 [TOOL] Simulating tool call...")
    
    query = state.get("query", "")
    
    # Simulate a tool call (like search or calculator)
    tool_call = {
        "tool": "search",
        "input": query,
        "result": f"Simulated result for: {query[:30]}..."
    }
    
    # Accumulate tool calls
    tool_calls = state.get("tool_calls", [])
    tool_calls.append(tool_call)
    state["tool_calls"] = tool_calls
    
    # Add tool result as a message
    tool_message = ToolMessage(
        content=f"Tool result: {tool_call['result']}",
        tool_call_id="simulated"
    )
    state["messages"] = state.get("messages", []) + [tool_message]
    
    # Accumulate intermediate results
    intermediate = state.get("intermediate_results", [])
    intermediate.append(f"Tool call {len(tool_calls)}: {tool_call['result']}")
    state["intermediate_results"] = intermediate
    
    state["status"] = "tool_complete"
    
    print(f"   Tool calls: {len(tool_calls)}")
    print(f"   Intermediate results: {len(intermediate)}")
    
    return state

def generate_final_response(state: AgentState) -> AgentState:
    """Generate the final response."""
    print("✅ [FINAL] Generating final response...")
    
    messages = state.get("messages", [])
    tool_calls = state.get("tool_calls", [])
    
    # Build summary of what happened
    summary = f"""
    Query: {state.get('query', '')}
    
    Tool calls made: {len(tool_calls)}
    
    Conversation turns: {len([m for m in messages if isinstance(m, HumanMessage)])}
    
    Final response: {state.get('current_response', '')}
    """
    
    state["current_response"] = summary
    state["status"] = "complete"
    
    print(f"   Final response length: {len(summary)}")
    print(f"   Status: {state['status']}")
    
    return state

# ============================================
# STEP 3: ROUTING FUNCTION
# ============================================

def should_continue(state: AgentState) -> Literal["call_llm", "simulate_tool", "final"]:
    """Decide what to do next based on state."""
    
    status = state.get("status", "")
    iteration = state.get("iteration_count", 0)
    messages = state.get("messages", [])
    
    # If we have a final response, stop
    if status == "complete":
        return "final"
    
    # If we've done 2 iterations, stop
    if iteration >= 2:
        return "final"
    
    # If we have a query but no LLM response yet
    if status == "processing":
        return "call_llm"
    
    # If we've called LLM and haven't simulated tool
    if status == "llm_complete":
        return "simulate_tool"
    
    # If we've simulated tool and haven't final
    if status == "tool_complete":
        return "final"
    
    return "final"

# ============================================
# STEP 4: BUILD THE GRAPH
# ============================================

def build_multi_turn_graph():
    """Build the multi-turn agent graph."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("process", process_query)
    graph.add_node("call_llm", call_llm)
    graph.add_node("simulate_tool", simulate_tool_call)
    graph.add_node("final", generate_final_response)
    
    # Set entry point
    graph.set_entry_point("process")
    
    # Add edges
    graph.add_edge("process", "call_llm")
    
    # Conditional routing
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "simulate_tool": "simulate_tool",
            "final": "final"
        }
    )
    
    graph.add_conditional_edges(
        "simulate_tool",
        should_continue,
        {
            "call_llm": "call_llm",  # Loop back
            "final": "final"
        }
    )
    
    graph.add_edge("final", END)
    
    return graph.compile()

# ============================================
# STEP 5: RUN DEMO
# ============================================

def run_demo():
    print("=" * 60)
    print("📝 MULTI-TURN STATEFUL AGENT")
    print("=" * 60)
    
    test_queries = [
        "What is RAG?",
        "Explain embeddings",
        "How does LangGraph work?"
    ]
    
    graph = build_multi_turn_graph()
    
    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"📝 Query: {query}")
        print("-" * 40)
        
        initial_state = {
            "messages": [],
            "query": query,
            "tool_calls": [],
            "intermediate_results": [],
            "current_response": "",
            "status": "pending",
            "iteration_count": 0
        }
        
        final_state = graph.invoke(initial_state)
        
        print("\n📊 Final State:")
        print(f"   Status: {final_state.get('status', 'unknown')}")
        print(f"   Iterations: {final_state.get('iteration_count', 0)}")
        print(f"   Tool calls: {len(final_state.get('tool_calls', []))}")
        print(f"   Intermediate results: {len(final_state.get('intermediate_results', []))}")
        print(f"   Total messages: {len(final_state.get('messages', []))}")
        
        print(f"\n   Final Response:\n{final_state.get('current_response', 'No response')[:200]}...")

if __name__ == "__main__":
    run_demo()
