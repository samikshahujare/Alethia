"""Automated unit and integration tests for LangGraph state machine orchestration.

Covers all Phase 4 requirements:
- Deterministic routing logic (route_after_judge)
- Scenario A: Supported question (1-cycle happy path, 0 rewrites)
- Scenario B: First failure then success (1 rewrite, 2 generation cycles)
- Scenario C: Two retries then success (2 rewrites, 3 generation cycles)
- Scenario D: Maximum retry limit (3 failures -> terminate after 2 rewrites)
- Scenario E: Proper abstention handling ("Not in context." is PASS, 0 rewrites)
- Scenario F: Empty retrieval context handling
- Scenario G: Trace integrity and step recording
- Scenario H: Original question preservation across multiple rewrites
- Error handling: Empty/whitespace query validation
"""

import pytest
from langgraph.graph import END

from src.graph import (
    MAX_REWRITES,
    build_rag_graph,
    route_after_judge,
    run_rag_pipeline,
)
from src.state import GraphState, create_initial_state


# =====================================================================
# Unit Tests for route_after_judge
# =====================================================================

class TestRouteAfterJudge:
    """Verify conditional edge routing rules based on verdict and attempt count."""

    def test_pass_verdict_always_routes_to_end(self):
        """A PASS verdict must always route to END, regardless of attempt count."""
        state_0: GraphState = {"verdict": "PASS", "attempts": 0}  # type: ignore
        state_1: GraphState = {"verdict": "PASS", "attempts": 1}  # type: ignore
        state_2: GraphState = {"verdict": "PASS", "attempts": 2}  # type: ignore
        state_lower: GraphState = {"verdict": "pass", "attempts": 0}  # type: ignore

        assert route_after_judge(state_0) == END
        assert route_after_judge(state_1) == END
        assert route_after_judge(state_2) == END
        assert route_after_judge(state_lower) == END

    def test_fail_verdict_with_attempts_under_max_routes_to_rewrite(self):
        """A FAIL verdict with attempts < 2 must route to 'rewrite'."""
        state_attempt_0: GraphState = {"verdict": "FAIL", "attempts": 0}  # type: ignore
        state_attempt_1: GraphState = {"verdict": "FAIL", "attempts": 1}  # type: ignore

        assert route_after_judge(state_attempt_0) == "rewrite"
        assert route_after_judge(state_attempt_1) == "rewrite"

    def test_fail_verdict_with_attempts_at_or_above_max_routes_to_end(self):
        """A FAIL verdict with attempts >= 2 must terminate at END."""
        state_attempt_2: GraphState = {"verdict": "FAIL", "attempts": 2}  # type: ignore
        state_attempt_3: GraphState = {"verdict": "FAIL", "attempts": 3}  # type: ignore

        assert route_after_judge(state_attempt_2) == END
        assert route_after_judge(state_attempt_3) == END

    def test_missing_or_empty_verdict_treated_as_fail(self):
        """Missing or malformed verdict is treated as FAIL for safe routing."""
        state_empty: GraphState = {"verdict": "", "attempts": 0}  # type: ignore
        state_none: GraphState = {"verdict": None, "attempts": 0}  # type: ignore

        assert route_after_judge(state_empty) == "rewrite"
        assert route_after_judge(state_none) == "rewrite"


# =====================================================================
# Integration Tests for build_rag_graph & run_rag_pipeline
# =====================================================================

class TestLangGraphScenarios:
    """Full workflow scenario tests using dependency-injected mock callables."""

    def test_scenario_a_supported_question_single_cycle(self):
        """Scenario A: Supported question passes in 1 cycle with 0 rewrites."""
        mock_chunks = [
            {"chunk_id": "c1", "source": "test.txt", "text": "The Transformer uses 6 layers."}
        ]

        def mock_retriever(query: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                return "The Transformer uses 6 layers."
            elif "strict fact-checker" in system_msg:
                return "VERDICT: PASS\nREASON: The draft is factually supported by the context."
            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("How many layers does the Transformer use?", graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert "6 layers" in final_state["draft"]
        assert final_state["original_question"] == "How many layers does the Transformer use?"
        assert final_state["question"] == "How many layers does the Transformer use?"

        # Verify trace node sequence
        node_names = [e.get("node") for e in final_state["trace"] if "node" in e]
        assert node_names == ["retrieve_node", "generate_node", "judge_node"]
        assert final_state["trace"][-1]["event"] == "workflow_completed"

    def test_scenario_b_first_failure_then_success(self):
        """Scenario B: First judgment fails; query is rewritten once; second judgment passes."""
        call_count = {"generate": 0, "judge": 0, "rewrite": 0, "retrieve": 0}

        def mock_retriever(query: str, k: int = 4):
            call_count["retrieve"] += 1
            if call_count["retrieve"] == 1:
                return [{"chunk_id": "c1", "source": "t.txt", "text": "General deep learning text."}]
            return [{"chunk_id": "c2", "source": "t.txt", "text": "Self-attention runs in O(n^2) time."}]

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                call_count["generate"] += 1
                if call_count["generate"] == 1:
                    return "Self-attention complexity is quadratic and uses multi-head fusion."
                return "Self-attention runs in O(n^2) computational complexity."

            elif "strict fact-checker" in system_msg:
                call_count["judge"] += 1
                if call_count["judge"] == 1:
                    return "VERDICT: FAIL\nREASON: Context mentions nothing about quadratic complexity or multi-head fusion."
                return "VERDICT: PASS\nREASON: Context explicitly confirms self-attention runs in O(n^2) time."

            elif "query reformulator" in system_msg:
                call_count["rewrite"] += 1
                return "What is the computational complexity of self-attention in the Transformer?"

            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("Tell me about attention speed.", graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 1
        assert call_count["generate"] == 2
        assert call_count["judge"] == 2
        assert call_count["rewrite"] == 1
        assert call_count["retrieve"] == 2

        # Trace checks
        node_names = [e.get("node") for e in final_state["trace"] if "node" in e]
        assert node_names == [
            "retrieve_node", "generate_node", "judge_node",
            "rewrite_node",
            "retrieve_node", "generate_node", "judge_node"
        ]
        assert final_state["original_question"] == "Tell me about attention speed."
        assert final_state["question"] == "What is the computational complexity of self-attention in the Transformer?"

    def test_scenario_c_two_retries_then_success(self):
        """Scenario C: Fails twice, rewritten twice (attempts=2), succeeds on 3rd cycle."""
        call_count = {"generate": 0, "judge": 0, "rewrite": 0}

        def mock_retriever(query: str, k: int = 4):
            return [{"chunk_id": "c1", "source": "t.txt", "text": "Adam optimizer is used with beta1=0.9."}]

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                call_count["generate"] += 1
                return f"Draft answer cycle {call_count['generate']}"

            elif "strict fact-checker" in system_msg:
                call_count["judge"] += 1
                if call_count["judge"] < 3:
                    return f"VERDICT: FAIL\nREASON: Hallucinated details in cycle {call_count['judge']}."
                return "VERDICT: PASS\nREASON: Completely supported by context."

            elif "query reformulator" in system_msg:
                call_count["rewrite"] += 1
                return f"Rewritten query {call_count['rewrite']}"

            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("Which optimizer is used?", graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 2
        assert call_count["generate"] == 3
        assert call_count["judge"] == 3
        assert call_count["rewrite"] == 2

    def test_scenario_d_maximum_retry_limit_exceeded(self):
        """Scenario D: Fails 3 consecutive times. Halts after 2 rewrites; no 4th generation."""
        call_count = {"generate": 0, "judge": 0, "rewrite": 0, "retrieve": 0}

        def mock_retriever(query: str, k: int = 4):
            call_count["retrieve"] += 1
            return [{"chunk_id": "c1", "source": "t.txt", "text": "Unrelated context."}]

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                call_count["generate"] += 1
                return f"Unverified claim cycle {call_count['generate']}"

            elif "strict fact-checker" in system_msg:
                call_count["judge"] += 1
                return f"VERDICT: FAIL\nREASON: Claim {call_count['judge']} unsupported."

            elif "query reformulator" in system_msg:
                call_count["rewrite"] += 1
                return f"Rewritten attempt {call_count['rewrite']}"

            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("Persistent unanswerable question", graph=graph)

        # Must terminate with FAIL after exactly MAX_REWRITES (2)
        assert final_state["verdict"] == "FAIL"
        assert final_state["attempts"] == MAX_REWRITES
        assert call_count["generate"] == 3
        assert call_count["judge"] == 3
        assert call_count["rewrite"] == 2
        assert call_count["retrieve"] == 3

        # Exactly 3 generation nodes, zero 4th generation
        gen_nodes = [e for e in final_state["trace"] if e.get("node") == "generate_node"]
        assert len(gen_nodes) == 3

        rew_nodes = [e for e in final_state["trace"] if e.get("node") == "rewrite_node"]
        assert len(rew_nodes) == 2

    def test_scenario_e_proper_abstention_is_pass(self):
        """Scenario E: 'Not in context.' for missing information is evaluated as PASS (0 rewrites)."""
        mock_chunks = [
            {"chunk_id": "c1", "source": "trans.txt", "text": "The Transformer uses multi-head attention."}
        ]

        def mock_retriever(query: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                # LLM correctly recognises quantum computing is absent from context
                return "Not in context."
            elif "strict fact-checker" in system_msg:
                # Judge verifies context does not contain quantum computing; valid abstention
                return "VERDICT: PASS\nREASON: Context does not contain quantum computing info; model correctly abstained."
            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("Does the Transformer use superconducting qubits?", graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["draft"] == "Not in context."
        assert final_state["attempts"] == 0
        # No rewrite occurred because abstention passed
        assert not any(e.get("node") == "rewrite_node" for e in final_state["trace"])

    def test_scenario_f_empty_retrieval_handling(self):
        """Scenario F: Empty retrieval context gracefully produces 'Not in context.' and PASS."""
        def mock_empty_retriever(query: str, k: int = 4):
            return []

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict fact-checker" in system_msg:
                return "VERDICT: PASS\nREASON: Empty context confirms model properly abstained."
            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_empty_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("Any random query with empty database", graph=graph)

        assert final_state["chunks"] == []
        assert final_state["draft"] == "Not in context."
        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0

    def test_scenario_g_trace_integrity(self):
        """Scenario G: Trace list records chronological metadata across all node executions."""
        mock_chunks = [{"chunk_id": "c1", "source": "doc1.txt", "text": "Some text."}]

        def mock_retriever(query: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                return "Answer text."
            elif "strict fact-checker" in system_msg:
                return "VERDICT: PASS\nREASON: All claims verified."
            return ""

        final_state = run_rag_pipeline(
            "What is text?",
            retriever_func=mock_retriever,
            llm_func=mock_llm,
        )

        trace = final_state["trace"]
        assert len(trace) >= 4  # retrieve, generate, judge, workflow_completed

        assert trace[0]["node"] == "retrieve_node"
        assert trace[0]["chunk_count"] == 1
        assert trace[0]["chunk_ids"] == ["c1"]

        assert trace[1]["node"] == "generate_node"
        assert trace[1]["draft"] == "Answer text."

        assert trace[2]["node"] == "judge_node"
        assert trace[2]["verdict"] == "PASS"

        last_entry = trace[-1]
        assert last_entry["event"] == "workflow_completed"
        assert last_entry["final_verdict"] == "PASS"
        assert last_entry["total_attempts"] == 0

    def test_scenario_h_original_question_preservation(self):
        """Scenario H: original_question remains invariant through multiple rewrites."""
        original_query = "What is the secret formula?"
        rewritten_queries = ["Formula question rev 1", "Formula question rev 2"]
        rewrite_idx = 0

        def mock_retriever(query: str, k: int = 4):
            return [{"chunk_id": "c1", "source": "d.txt", "text": "Nothing."}]

        def mock_llm(messages, temperature=0.0, max_tokens=1024):
            nonlocal rewrite_idx
            system_msg = messages[0]["content"]
            if "strict, factual assistant" in system_msg:
                return "Formula is X"
            elif "strict fact-checker" in system_msg:
                return "VERDICT: FAIL\nREASON: Unsupported"
            elif "query reformulator" in system_msg:
                res = rewritten_queries[rewrite_idx]
                rewrite_idx += 1
                return res
            return ""

        final_state = run_rag_pipeline(
            original_query,
            retriever_func=mock_retriever,
            llm_func=mock_llm,
        )

        assert final_state["original_question"] == original_query
        assert final_state["question"] == rewritten_queries[-1]
        assert final_state["attempts"] == 2


# =====================================================================
# Error Handling Tests
# =====================================================================

class TestErrorHandling:
    """Validate query input sanitization and error handling."""

    def test_empty_question_raises_value_error(self):
        with pytest.raises(ValueError, match="Question cannot be empty"):
            run_rag_pipeline("")

    def test_whitespace_question_raises_value_error(self):
        with pytest.raises(ValueError, match="Question cannot be empty"):
            run_rag_pipeline("   \t\n  ")

    def test_none_question_raises_value_error(self):
        with pytest.raises(ValueError, match="Question cannot be empty"):
            run_rag_pipeline(None)  # type: ignore

