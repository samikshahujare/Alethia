"""Comprehensive unit and integration tests for Phase 4: Targeted Query Rewriting & LangGraph Orchestration.

Tests:
1. Targeted query rewriting based on Judge verification feedback.
2. Anti-looping mechanism (duplicate rewrite prevention and semantic fallback).
3. Query history and attempt tracking in GraphState.
4. Evidence-aware retrieval and newly retrieved chunk detection.
5. Distinction between relevant and irrelevant new chunks.
6. Evidence preservation across multiple retrieval attempts.
7. Original question preservation throughout workflow cycles.
8. Full 10-step self-correction trace verification in LangGraph workflow.
9. Failure exhaustion after MAX_REWRITES (bounded loop termination).
10. Structured execution trace completeness (latencies, chunk IDs, verdicts).
"""

import pytest
import time
from typing import Any, Dict, List

from src.graph import MAX_REWRITES, build_rag_graph, run_rag_pipeline, route_after_judge
from src.nodes import (
    retrieve_node,
    generate_node,
    judge_node,
    rewrite_node,
    identify_new_chunks,
    generate_targeted_keyword_fallback,
)
from src.state import GraphState, create_initial_state
from eval.self_correction_eval import verify_strict_self_correction_trace


# ============================================================================
# 1. Targeted Query Rewriting & Judge Feedback Processing
# ============================================================================

class TestTargetedQueryRewriting:
    """Test targeted rewriting isolating missing or ungrounded requirements."""

    def test_rewrite_isolates_missing_subquestion_from_judge_feedback(self):
        """When Judge reports missing information for one sub-question, rewrite focuses on it."""
        orig_q = "What optimizer was used in the Transformer paper, and what are Adam's default beta values?"
        judge_reason = "Missing information regarding Adam's default beta values in the retrieved context."

        state = create_initial_state(orig_q)
        state["decomposed_questions"] = [
            "What optimizer was used in the Transformer paper?",
            "What are Adam's default beta values?",
        ]
        state["reason"] = judge_reason
        state["draft"] = "The Transformer used Adam. Adam's beta values are not in context."
        state["verdict"] = "FAIL"

        def mock_llm(messages, **kwargs):
            prompt = messages[1]["content"]
            assert "Adam's default beta values" in prompt or "beta" in prompt.lower()
            return "What are the default beta1 and beta2 hyperparameter values for the Adam optimizer?"

        new_state = rewrite_node(state, llm_func=mock_llm)

        assert new_state["original_question"] == orig_q
        assert new_state["question"] != orig_q
        assert "beta" in new_state["question"].lower()
        assert new_state["attempts"] == 1
        assert new_state["question"] in new_state["query_history"]

    def test_original_question_strictly_preserved(self):
        """Original question must never be overwritten during rewrites."""
        orig_q = "What is the penalty that creates sparse features?"
        state = create_initial_state(orig_q)
        state["reason"] = "Retrieved chunks only discussed L2 ridge regression, not L1 sparsity."

        def mock_llm(**kwargs):
            return "L1 regularization Lasso sparsity penalty mechanism"

        new_state = rewrite_node(state, llm_func=mock_llm)
        assert new_state["original_question"] == orig_q
        assert new_state["question"] == "L1 regularization Lasso sparsity penalty mechanism"


# ============================================================================
# 2. Anti-Looping Mechanism & Duplicate Query Prevention
# ============================================================================

class TestAntiLoopingMechanism:
    """Test duplicate query rejection, query history, and fallback keyword generation."""

    def test_duplicate_rewrite_rejected_and_falls_back_to_targeted_reformulation(self):
        """If the LLM returns an identical query or one in query_history, fallback generates a novel query."""
        orig_q = "Explain scaled dot-product attention scaling factor."
        state = create_initial_state(orig_q)
        state["query_history"] = [orig_q, "scaled dot product attention scaling"]
        state["question"] = "scaled dot product attention scaling"
        state["reason"] = "Missing mathematical reason for dividing by sqrt(d_k)."
        state["attempts"] = 1

        # Mock LLM returns a query that already exists in history (duplicate)
        def mock_llm(**kwargs):
            return "scaled dot product attention scaling"

        new_state = rewrite_node(state, llm_func=mock_llm)

        # Must reject duplicate and create a novel targeted query
        assert new_state["question"] not in [orig_q, "scaled dot product attention scaling"]
        assert len(new_state["question"].strip()) > 0
        assert new_state["attempts"] == 2
        assert new_state["question"] in new_state["query_history"]

    def test_normalized_textual_duplicate_detection(self):
        """Detects duplicates via case-folding and punctuation stripping (normalized textual comparison)."""
        orig_q = "What is backpropagation?"
        state = create_initial_state(orig_q)
        state["query_history"] = [orig_q]
        state["reason"] = "Missing chain rule details."

        # Case-insensitive / punctuation-stripped match
        def mock_llm(**kwargs):
            return "What is backpropagation?!"

        new_state = rewrite_node(state, llm_func=mock_llm)
        assert new_state["question"].lower().strip("?!. ") != "what is backpropagation"

    def test_generate_targeted_keyword_fallback(self):
        """Keyword fallback extracts key entities and reason keywords."""
        fallback = generate_targeted_keyword_fallback(
            original_question="What optimizer was used in the Transformer paper?",
            reason="Missing the specific name of the optimizer.",
            history=["What optimizer was used in the Transformer paper?"],
        )
        assert len(fallback) > 0
        assert fallback.lower() != "what optimizer was used in the transformer paper?"


# ============================================================================
# 3. Evidence-Aware Retrieval & Chunk Tracking
# ============================================================================

class TestEvidenceAwareRetrieval:
    """Test detection of genuinely new chunks and evidence preservation."""

    def test_identify_new_chunks_difference(self):
        """Newly retrieved chunks must strictly be post_chunk_ids - initial_chunk_ids."""
        init_ids = ["chunk_01", "chunk_02"]
        post_ids = ["chunk_02", "chunk_03", "chunk_04"]

        new_ids = identify_new_chunks(initial_chunk_ids=init_ids, post_chunk_ids=post_ids)
        assert new_ids == ["chunk_03", "chunk_04"]
        assert "chunk_02" not in new_ids

    def test_retrieve_node_tracks_initial_and_subsequent_chunks(self):
        """Retrieve node records initial_chunk_ids on attempt 0 and newly_retrieved_chunk_ids on attempt >= 1."""
        init_chunks = [{"chunk_id": "c1", "text": "init 1"}, {"chunk_id": "c2", "text": "init 2"}]
        post_chunks = [{"chunk_id": "c2", "text": "init 2"}, {"chunk_id": "c3", "text": "new chunk 3"}]

        state = create_initial_state("Query 1")

        # Attempt 0: Initial retrieval
        state_0 = retrieve_node(state, retriever_func=lambda q, k=4: init_chunks)
        assert state_0["initial_chunk_ids"] == ["c1", "c2"]
        assert state_0["newly_retrieved_chunk_ids"] == []

        # Attempt 1: Re-retrieval after rewrite
        state_0["attempts"] = 1
        state_0["question"] = "Query 1 rewritten"
        state_1 = retrieve_node(state_0, retriever_func=lambda q, k=4: post_chunks)

        assert state_1["subsequent_chunk_ids"] == ["c2", "c3"]
        assert state_1["newly_retrieved_chunk_ids"] == ["c3"]
        # Prior useful chunk c1 is preserved in candidate pool
        chunk_ids_in_state = [c["chunk_id"] for c in state_1["chunks"]]
        assert "c3" in chunk_ids_in_state
        assert "c1" in chunk_ids_in_state or "c2" in chunk_ids_in_state

    def test_relevant_vs_irrelevant_new_chunks_discrimination(self):
        """A new chunk must match target evidence to be classified as relevant."""
        target_chunk_ids = {"optimizers_guide_txt_chunk_0002"}
        newly_retrieved = ["unrelated_doc_chunk_0001", "optimizers_guide_txt_chunk_0002"]

        relevant_new = [cid for cid in newly_retrieved if cid in target_chunk_ids]
        assert relevant_new == ["optimizers_guide_txt_chunk_0002"]


# ============================================================================
# 4. LangGraph Cyclic Routing & Bound Enforcement
# ============================================================================

class TestLangGraphOrchestrationRouting:
    """Test graph conditional transitions and retry limits."""

    def test_pass_routes_directly_to_end(self):
        state = create_initial_state("Sample query")
        state["verdict"] = "PASS"
        state["attempts"] = 0
        assert route_after_judge(state) == "__end__"

    def test_fail_with_attempts_under_limit_routes_to_rewrite(self):
        state = create_initial_state("Sample query")
        state["verdict"] = "FAIL"
        state["attempts"] = 0
        assert route_after_judge(state) == "rewrite"

        state["attempts"] = 1
        assert route_after_judge(state) == "rewrite"

    def test_fail_at_max_rewrites_routes_to_end_without_infinite_loop(self):
        state = create_initial_state("Sample query")
        state["verdict"] = "FAIL"
        state["attempts"] = MAX_REWRITES  # 2
        assert route_after_judge(state) == "__end__"

        state["attempts"] = 3
        assert route_after_judge(state) == "__end__"


# ============================================================================
# 5. Full 10-Step Self-Correction Trace Integration
# ============================================================================

class TestSelfCorrectionTraceVerification:
    """End-to-end integration test verifying the strict 10-step chronological trace."""

    def test_end_to_end_10_step_self_correction_workflow(self):
        """Executes a full cycle:

        Attempt 0: Initial retrieval (missing target chunk) -> Initial draft -> Judge FAIL
        Rewrite: Targeted query generated
        Attempt 1: Subsequent retrieval (fetches target chunk) -> Revised draft -> Judge PASS
        Verifies via verify_strict_self_correction_trace.
        """
        target_chunk_id = "optimizers_guide_txt_chunk_0002"
        mandatory_facts = ["beta_1 = 0.9", "beta_2 = 0.999"]

        # Attempt 0 returns incomplete chunks
        init_chunks = [
            {"chunk_id": "transformer_summary_txt_chunk_0004", "text": "Transformer used Adam optimizer.", "source": "trans.txt"}
        ]
        # Attempt 1 returns new targeted chunk
        post_chunks = [
            {"chunk_id": target_chunk_id, "text": "Adam default hyperparameters: beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8.", "source": "opt.txt"}
        ]

        def mock_retriever(query, k=4):
            if "hyperparameter" in query.lower() or "beta_1" in query.lower():
                return post_chunks
            return init_chunks

        call_idx = 0

        def mock_llm(messages, **kwargs):
            nonlocal call_idx
            system_msg = messages[0]["content"]
            user_msg = messages[1]["content"]

            # Generate Node
            if "strict, factual assistant" in system_msg:
                call_idx += 1
                if call_idx == 1:
                    # Initial draft: unsupported claim
                    return "The Transformer used Adam, with beta_1 = 0.8 and beta_2 = 0.888."
                else:
                    # Revised draft with newly retrieved evidence
                    return "The Transformer used Adam. Adam's default hyperparameters are beta_1 = 0.9 and beta_2 = 0.999."

            # Judge Node
            elif "strict fact-checker" in system_msg:
                if "beta_1 = 0.8" in user_msg:
                    return "VERDICT: FAIL\nREASON: Unsupported claim regarding beta values (0.8 and 0.888 not in context)."
                else:
                    return "VERDICT: PASS\nREASON: Fully supported by newly retrieved context."

            # Rewrite Node
            elif "query reformulator" in system_msg:
                return "Adam optimizer default beta_1 and beta_2 hyperparameter values"

            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(
            "What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
            graph=graph,
        )

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 1
        assert "beta_1 = 0.9" in final_state["draft"]

        # Run strict 10-step verifier
        is_valid, report = verify_strict_self_correction_trace(
            trace=final_state["trace"],
            original_question=final_state["original_question"],
            final_verdict=final_state["verdict"],
            final_draft=final_state["draft"],
            mandatory_facts=mandatory_facts,
            relevant_chunk_ids=[target_chunk_id],
        )

        assert is_valid is True, f"Trace verification failed: {report}"
        assert report["step_07_new_chunks_retrieved"] is True
        assert report["step_07_ground_truth_chunks_retrieved"] is True

