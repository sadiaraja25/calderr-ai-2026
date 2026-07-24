# Week 4: LangGraph, Workflows & State Machines

## Overview
This week focused on graph-based agent orchestration using LangGraph.

## Daily Breakdown

### Monday: LangGraph Foundations
- First graph with State → Node → Edge model
- Document processing with conditional routing

### Tuesday: Branching & Loops
- Classification router (general/technical/sensitive)
- Self-correcting agent loop (max 3 retries)

### Wednesday: Stateful Agents
- Multi-turn agent with accumulated state
- Research workflow with quality threshold

### Thursday: Human-in-the-Loop
- Approval workflow (propose → review → execute)
- Content moderation with persistence

### Friday: Production Graphs
- LangSmith tracing
- SqliteSaver persistence
- Interrupt and resume capabilities

## Labs
- Lab 4.1: Document Processing Graph
- Lab 4.2: Self-Correcting Agent Loop
- Lab 4.3: Human-in-the-Loop Approval Workflow

## Key Concepts Learned
- LangGraph StateGraph
- Conditional edges and routing
- TypedDict state schemas
- Annotated reducers
- Human-in-the-loop patterns
- State persistence with SqliteSaver
