"""LangGraph state machine orchestration for Alethia.

Constructs and compiles the cyclic self-correcting RAG workflow:
START -> retrieve -> generate -> judge
After judge:
- PASS -> END
- FAIL (attempts < 2) -> rewrite -> retrieve
- FAIL (attempts >= 2) -> END
"""

import time
from typing import Any, Callable, Dict, List, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.nodes import generate_node, judge_node, retrieve_node, rewrite_node
from src.state import GraphState, create_initial_state

MAX_REWRITES = 2


def route_after_judge(state: GraphState) -> str:
    """Determine the next step after the Judge node evaluations.

    Rules:
    - If verdict is PASS: Route to END.
    - If verdict is FAIL and attempts < 2: Route to 'rewrite'.
    - If verdict is FAIL and attempts >= 2: Route to END (max retries reached).

    Args:
        state: Current GraphState.

    Returns:
        Next node name ('rewrite') or END string.
    """
    verdict = (state.get("verdict") or "").strip().upper()
    attempts = state.get("attempts", 0)

    if verdict == "PASS":
        return END

    if attempts < MAX_REWRITES:
        return "rewrite"

    return END


def build_rag_graph(
    retriever_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    llm_func: Optional[Callable[..., str]] = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph self-correcting RAG StateGraph.

    Supports optional dependency injection for retriever and LLM functions
    to enable fully deterministic, zero-cost unit testing.

    Args:
        retriever_func: Optional custom retrieval callable.
        llm_func: Optional custom LLM callable.

    Returns:
        CompiledStateGraph instance ready for invocation.
    """
    workflow = StateGraph(GraphState)

    # Wrap node functions with injected callables if provided
    def _retrieve(s: GraphState) -> GraphState:
        return retrieve_node(s, retriever_func=retriever_func)

    def _generate(s: GraphState) -> GraphState:
        return generate_node(s, llm_func=llm_func)

    def _judge(s: GraphState) -> GraphState:
        return judge_node(s, llm_func=llm_func)

    def _rewrite(s: GraphState) -> GraphState:
        return rewrite_node(s, llm_func=llm_func)

    # Register nodes
    workflow.add_node("retrieve", _retrieve)
    workflow.add_node("generate", _generate)
    workflow.add_node("judge", _judge)
    workflow.add_node("rewrite", _rewrite)

    # Set up fixed linear edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "judge")

    # Set up conditional routing from judge
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            END: END,
            "rewrite": "rewrite",
        },
    )

    # Loop back from rewrite to retrieve
    workflow.add_edge("rewrite", "retrieve")

    return workflow.compile()


# Global cached compiled graph for default production execution
_DEFAULT_GRAPH: Optional[CompiledStateGraph] = None


def get_compiled_graph() -> CompiledStateGraph:
    """Retrieve or build the default production compiled graph."""
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = build_rag_graph()
    return _DEFAULT_GRAPH


def run_rag_pipeline(
    question: str,
    graph: Optional[CompiledStateGraph] = None,
    retriever_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    llm_func: Optional[Callable[..., str]] = None,
) -> GraphState:
    """Execute the complete self-correcting RAG workflow for a user question.

    Args:
        question: User query string.
        graph: Optional pre-compiled LangGraph instance.
        retriever_func: Optional custom retriever function (used if building a graph).
        llm_func: Optional custom LLM function (used if building a graph).

    Returns:
        Final GraphState dictionary after graph termination.

    Raises:
        ValueError: If the question is empty or whitespace only.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        raise ValueError("Question cannot be empty or whitespace only.")

    start_time = time.time()
    initial_state = create_initial_state(clean_question)

    active_graph = graph or (
        build_rag_graph(retriever_func=retriever_func, llm_func=llm_func)
        if (retriever_func or llm_func)
        else get_compiled_graph()
    )

    final_state = active_graph.invoke(initial_state)
    total_latency_ms = round((time.time() - start_time) * 1000, 2)
    final_state["total_latency_ms"] = total_latency_ms

    # Append final completion record to execution trace
    trace_list = list(final_state.get("trace", []))
    trace_list.append({
        "event": "workflow_completed",
        "final_verdict": final_state.get("verdict"),
        "total_attempts": final_state.get("attempts"),
        "total_trace_steps": len(trace_list),
        "total_latency_ms": total_latency_ms,
        "initial_chunk_ids": final_state.get("initial_chunk_ids", []),
        "subsequent_chunk_ids": final_state.get("subsequent_chunk_ids", []),
        "newly_retrieved_chunk_ids": final_state.get("newly_retrieved_chunk_ids", []),
    })
    final_state["trace"] = trace_list

    return final_state
