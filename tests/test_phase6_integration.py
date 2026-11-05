"""Comprehensive end-to-end integration and regression test suite for Phase 6.

Validates all 15 core architectural and integration scenarios for Alethia:
1. Simple answerable question (1 cycle, attempt 0, PASS).
2. Multi-intent answerable question (decomposition, independent retrieval, RRF fusion, attempt 0, PASS).
3. Correct abstention for missing information (Rule 2 partial or full abstention, attempt 0, PASS).
4. Out-of-scope question (unrelated topic, "Not in context.", Rule 2 PASS).
5. Partially answerable composite question (Part A answered, Part B cleanly abstained, Rule 2 PASS).
6. Unsupported factual claim (hallucinated draft contradicts/unsupported by context, Rule 3 FAIL).
7. Initial Judge FAIL followed by rewrite (cyclic self-correction, attempts=1, PASS).
8. Maximum retry exhaustion (2 rewrites executed, terminates cleanly at END with FAIL).
9. Duplicate rewrite prevention (anti-looping fallback triggered, distinct query produced).
10. Empty retrieval result (empty chunks detected, LLM bypassed, "Not in context.", PASS).
11. Retrieval service failure (retriever exception caught safely, no unhandled crash).
12. LLM API failure (LLM exception handled safely by pipeline/UI, graceful recovery).
13. Source attribution integrity (chunk_id, source, text, dense_rank, bm25_rank, rrf_score preserved).
14. Session history integrity (multi-turn conversation state preservation without leakage).
15. Document upload and indexing (file validation, chunking, deduplication, and BM25 synchronization).
"""

import os
from io import BytesIO
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
import streamlit as st

from src.config import is_groq_configured
from src.decomposition import (
    decompose_question,
    is_multi_intent_question,
    merge_subquestion_chunks,
    retrieve_decomposed,
)
from src.graph import MAX_REWRITES, build_rag_graph, run_rag_pipeline
from src.llm import ConfigurationError, LLMError, LLMRateLimitError
from src.nodes import (
    generate_node,
    generate_targeted_keyword_fallback,
    identify_new_chunks,
    judge_node,
    retrieve_node,
    rewrite_node,
)
from src.state import GraphState, create_initial_state
from ui.chat import init_session_state, process_user_query
from ui.sidebar import handle_document_upload
from ui.sources import render_sources
from ui.trace import render_trace


# ============================================================================
# Scenario 1: Simple Answerable Question
# ============================================================================

class TestScenario01SimpleAnswerable:
    """Scenario 1: Single-intent question with grounded context passes on attempt 0."""

    def test_simple_answerable_question_single_cycle_pass(self):
        question = "How many layers are in the Transformer encoder?"
        mock_chunks = [
            {
                "chunk_id": "transformer_chunk_001",
                "source": "transformer_summary.txt",
                "text": "The encoder is composed of a stack of N = 6 identical layers.",
                "dense_rank": 1,
                "bm25_rank": 1,
                "rrf_score": 0.032787,
            }
        ]

        def mock_retriever(q: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                return "The Transformer encoder is composed of 6 identical layers."
            elif "strict fact-checker" in sys_msg:
                return "VERDICT: PASS\nREASON: Factually supported by the retrieved context specifying N = 6 layers."
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(question, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert "6 identical layers" in final_state["draft"]
        assert len(final_state["chunks"]) == 1
        assert final_state["chunks"][0]["chunk_id"] == "transformer_chunk_001"
        assert final_state["original_question"] == question


# ============================================================================
# Scenario 2: Multi-Intent Answerable Question
# ============================================================================

class TestScenario02MultiIntentAnswerable:
    """Scenario 2: Multi-intent question decomposed, independently retrieved, and passed."""

    def test_multi_intent_decomposition_and_rrf_retrieval(self):
        composite_q = (
            "What optimizer was used in the Transformer paper, and what are Adam's default beta values?"
        )
        assert is_multi_intent_question(composite_q) is True

        # Independent evidence for each sub-question
        sub1 = "What optimizer was used in the Transformer paper?"
        sub2 = "What are Adam's default beta values?"

        chunks_sub1 = [
            {
                "chunk_id": "trans_opt_001",
                "source": "transformer_summary.txt",
                "text": "The authors used the Adam optimizer with beta1=0.9 and beta2=0.98.",
                "rrf_score": 0.032,
            }
        ]
        chunks_sub2 = [
            {
                "chunk_id": "adam_guide_001",
                "source": "optimizers_guide.txt",
                "text": "Adam's standard default hyperparameters are beta1 = 0.9 and beta2 = 0.999.",
                "rrf_score": 0.031,
            }
        ]

        sub_results = {sub1: chunks_sub1, sub2: chunks_sub2}
        merged = merge_subquestion_chunks(sub_results, k=4)
        merged_ids = [c["chunk_id"] for c in merged]
        assert "trans_opt_001" in merged_ids
        assert "adam_guide_001" in merged_ids

        # End-to-end pipeline execution with mock LLM
        def mock_retriever(q: str, k: int = 4):
            return merged

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                return (
                    "The Transformer paper used the Adam optimizer (with beta1=0.9, beta2=0.98). "
                    "In general, Adam's standard default values are beta1=0.9 and beta2=0.999."
                )
            elif "strict fact-checker" in sys_msg:
                user_msg = messages[1]["content"]
                assert composite_q in user_msg
                return "VERDICT: PASS\nREASON: Both sub-questions are completely addressed and grounded in the retrieved context."
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(composite_q, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert "Adam" in final_state["draft"]


# ============================================================================
# Scenario 3: Correct Abstention for Missing Information
# ============================================================================

class TestScenario03CorrectAbstention:
    """Scenario 3: Question where information is missing and generator correctly abstains (Rule 2 PASS)."""

    def test_correct_abstention_receives_pass(self):
        question = "What was the batch size used for pretraining BERT-Large in the original paper?"
        mock_chunks = [
            {
                "chunk_id": "nlp_chunk_001",
                "source": "nlp_concepts.txt",
                "text": "Byte-pair encoding (BPE) and SentencePiece are common subword tokenization algorithms.",
            }
        ]

        def mock_retriever(q: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                return "Not in context. The retrieved text only discusses tokenization algorithms, not BERT batch size."
            elif "strict fact-checker" in sys_msg:
                return "VERDICT: PASS\nREASON: The generator correctly abstains because the retrieved context does not contain the requested batch size."
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(question, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert "not in context" in final_state["draft"].lower()


# ============================================================================
# Scenario 4: Out-of-Scope Question
# ============================================================================

class TestScenario04OutOfScope:
    """Scenario 4: Query entirely outside corpus scope triggers clean abstention and PASS."""

    def test_out_of_scope_query_abstains_cleanly(self):
        question = "What is the capital city of Australia?"

        def mock_retriever(q: str, k: int = 4):
            # Out of scope query retrieves no relevant content
            return []

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict fact-checker" in sys_msg:
                return "VERDICT: PASS\nREASON: Information is absent from context and the generator correctly abstained."
            return "Unexpected"

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(question, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert final_state["draft"] == "Not in context."


# ============================================================================
# Scenario 5: Partially Answerable Composite Question
# ============================================================================

class TestScenario05PartiallyAnswerableComposite:
    """Scenario 5: Composite query where Part 1 is in context and Part 2 is missing."""

    def test_partially_answerable_composite_question_rule2_pass(self):
        composite_q = (
            "What optimizer was used in the Transformer paper, and what was the author's shoe size?"
        )
        mock_chunks = [
            {
                "chunk_id": "trans_001",
                "source": "transformer_summary.txt",
                "text": "The authors used the Adam optimizer with beta1=0.9 and beta2=0.98.",
            }
        ]

        def mock_retriever(q: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                return (
                    "The Transformer paper used the Adam optimizer. "
                    "Information regarding the author's shoe size is not in context."
                )
            elif "strict fact-checker" in sys_msg:
                user_msg = messages[1]["content"]
                assert composite_q in user_msg
                return (
                    "VERDICT: PASS\n"
                    "REASON: The generator correctly answered the supported optimizer portion and "
                    "properly abstained on the missing author shoe size without inventing facts."
                )
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(composite_q, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 0
        assert "Adam" in final_state["draft"]
        assert "not in context" in final_state["draft"].lower()


# ============================================================================
# Scenario 6: Unsupported Factual Claim (Rule 3 FAIL)
# ============================================================================

class TestScenario06UnsupportedClaim:
    """Scenario 6: Generator hallucinates an unsupported factual claim, receiving FAIL."""

    def test_unsupported_factual_claim_receives_fail(self):
        question = "What is the height of Olympus Mons on Mars?"
        mock_chunks = [
            {
                "chunk_id": "solar_001",
                "source": "solar_system.txt",
                "text": "Olympus Mons on Mars is the tallest planetary shield volcano in the Solar System.",
            }
        ]

        def mock_retriever(q: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                # Hallucinated exact height not in the retrieved chunk
                return "Olympus Mons is exactly 88,000 meters high and erupted in 1942."
            elif "strict fact-checker" in sys_msg:
                return (
                    "VERDICT: FAIL\n"
                    "REASON: The claim of 88,000 meters and eruption year 1942 is completely unsupported by context."
                )
            elif "query reformulation" in sys_msg:
                return "Olympus Mons shield volcano elevation height in kilometers"
            return ""

        state = create_initial_state(question)
        state["chunks"] = mock_chunks
        state["draft"] = "Olympus Mons is exactly 88,000 meters high and erupted in 1942."

        judged_state = judge_node(state, llm_func=mock_llm)
        assert judged_state["verdict"] == "FAIL"
        assert "unsupported" in judged_state["reason"].lower()


# ============================================================================
# Scenario 7: Initial Judge FAIL Followed by Targeted Rewrite (Self-Correction)
# ============================================================================

class TestScenario07InitialFailFollowedByRewrite:
    """Scenario 7: Full cyclic self-correction (Attempt 0 FAIL -> rewrite -> Attempt 1 PASS)."""

    def test_initial_fail_followed_by_successful_correction(self):
        orig_q = "What optimizer was used in the Transformer paper, and what is Adam's second-moment update formula?"

        initial_chunks = [
            {
                "chunk_id": "trans_001",
                "source": "transformer_summary.txt",
                "text": "The authors used the Adam optimizer in the Transformer architecture.",
                "dense_rank": 1,
                "bm25_rank": 1,
                "rrf_score": 0.032,
            }
        ]

        post_rewrite_chunks = [
            {
                "chunk_id": "opt_guide_000",
                "source": "optimizers_guide.txt",
                "text": "Adam computes the second raw moment estimate as v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2.",
                "dense_rank": 1,
                "bm25_rank": 1,
                "rrf_score": 0.032,
            }
        ]

        call_counts = {"retriever": 0, "generate": 0, "judge": 0, "rewrite": 0}

        def mock_retriever(query: str, k: int = 4):
            call_counts["retriever"] += 1
            if call_counts["retriever"] == 1:
                return initial_chunks
            return post_rewrite_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                call_counts["generate"] += 1
                if call_counts["generate"] == 1:
                    # Incomplete attempt 0
                    return "The Transformer uses Adam. The second moment formula is v_t = g_t."
                else:
                    # Grounded attempt 1
                    return (
                        "The Transformer paper used Adam. Adam's second-moment update formula is "
                        "v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2."
                    )
            elif "strict fact-checker" in sys_msg:
                call_counts["judge"] += 1
                if call_counts["judge"] == 1:
                    return (
                        "VERDICT: FAIL\n"
                        "REASON: The claim v_t = g_t is unsupported and contradicts standard momentum mathematics."
                    )
                else:
                    return (
                        "VERDICT: PASS\n"
                        "REASON: The answer is now completely grounded in the retrieved optimizers_guide context."
                    )
            elif "reformulat" in sys_msg.lower():
                call_counts["rewrite"] += 1
                return "Adam second-moment update formula v_t beta2 squared gradient"
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(orig_q, graph=graph)

        assert final_state["verdict"] == "PASS"
        assert final_state["attempts"] == 1
        assert final_state["original_question"] == orig_q
        assert call_counts["rewrite"] == 1
        assert call_counts["generate"] == 2
        assert call_counts["judge"] == 2
        assert "opt_guide_000" in final_state["newly_retrieved_chunk_ids"]
        assert len(final_state["chunks"]) == 2  # Evidence preservation: trans_001 + opt_guide_000


# ============================================================================
# Scenario 8: Maximum Retry Exhaustion
# ============================================================================

class TestScenario08MaxRetryExhaustion:
    """Scenario 8: Continuous FAIL verdicts exhaust MAX_REWRITES (2) and terminate at END."""

    def test_maximum_retry_exhaustion_terminates_at_end(self):
        question = "Explain quantum gravity in terms of Transformer attention."

        def mock_retriever(query: str, k: int = 4):
            return [{"chunk_id": "c1", "source": "s.txt", "text": "Transformer attention info."}]

        call_counts = {"generate": 0, "judge": 0, "rewrite": 0}

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                call_counts["generate"] += 1
                return f"Draft attempt {call_counts['generate']} about quantum gravity."
            elif "strict fact-checker" in sys_msg:
                call_counts["judge"] += 1
                return "VERDICT: FAIL\nREASON: Quantum gravity is completely ungrounded in Transformer text."
            elif "reformulat" in sys_msg.lower():
                call_counts["rewrite"] += 1
                return f"quantum gravity rewrite {call_counts['rewrite']}"
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline(question, graph=graph)

        assert final_state["verdict"] == "FAIL"
        assert final_state["attempts"] == MAX_REWRITES  # 2 rewrites
        assert call_counts["rewrite"] == 2
        assert call_counts["generate"] == 3  # Initial + 2 after rewrites
        assert call_counts["judge"] == 3


# ============================================================================
# Scenario 9: Duplicate Rewrite Prevention (Anti-Looping)
# ============================================================================

class TestScenario09DuplicateRewritePrevention:
    """Scenario 9: Rejection of repetitive/looping rewrites triggers targeted fallback."""

    def test_duplicate_rewrite_prevention_triggers_fallback(self):
        orig_q = "Explain scaled dot-product attention scaling factor."
        state = create_initial_state(orig_q)
        state["query_history"] = [orig_q, "scaled dot product attention scaling"]
        state["question"] = "scaled dot product attention scaling"
        state["reason"] = "Missing mathematical reason for dividing by sqrt(d_k)."
        state["attempts"] = 1

        # Mock LLM returns a query that already exists in history (duplicate)
        def mock_llm(**kwargs):
            return "scaled dot product attention scaling"

        rewritten_state = rewrite_node(state, llm_func=mock_llm)

        assert rewritten_state["question"] != "scaled dot product attention scaling"
        assert rewritten_state["question"] != orig_q
        assert rewritten_state["question"] in rewritten_state["query_history"]
        assert len(rewritten_state["query_history"]) == 3


# ============================================================================
# Scenario 10: Empty Retrieval Result
# ============================================================================

class TestScenario10EmptyRetrieval:
    """Scenario 10: Empty retrieval bypasses generator LLM and yields verified abstention."""

    def test_empty_retrieval_bypasses_generator_llm(self):
        question = "What is the orbital period of Neptune?"
        state = create_initial_state(question)
        state["chunks"] = []

        llm_called = False

        def mock_llm(**kwargs):
            nonlocal llm_called
            llm_called = True
            return "Should not be called"

        updated_state = generate_node(state, llm_func=mock_llm)

        assert llm_called is False
        assert updated_state["draft"] == "Not in context."
        trace_last = updated_state["trace"][-1]
        assert "Empty retrieval context; bypassed LLM generation." in trace_last.get("note", "")


# ============================================================================
# Scenario 11: Retrieval Service Failure
# ============================================================================

class TestScenario11RetrievalServiceFailure:
    """Scenario 11: Retriever exception is caught gracefully without pipeline crash."""

    def test_retriever_failure_handled_gracefully(self):
        question = "What is gradient clipping?"

        def failing_retriever(q: str, k: int = 4):
            raise RuntimeError("ChromaDB connection timeout (simulated service failure)")

        state = create_initial_state(question)
        # retrieve_node must catch the exception and fall back to empty chunks
        new_state = retrieve_node(state, retriever_func=failing_retriever)

        assert new_state["chunks"] == []
        assert new_state["initial_chunk_ids"] == []
        trace_entry = new_state["trace"][-1]
        assert trace_entry["chunk_count"] == 0


# ============================================================================
# Scenario 12: LLM API Failure Handling
# ============================================================================

class TestScenario12LLMAPIFailure:
    """Scenario 12: LLM API failures caught gracefully by UI and pipeline wrappers."""

    def test_llm_api_failure_handled_in_ui_process_query(self):
        # Reset session state
        st.session_state.messages = []
        init_session_state()

        def failing_pipeline(q: str):
            raise LLMError("Groq 503 Service Unavailable")

        with patch("ui.chat.is_groq_configured", return_value=True), patch("streamlit.spinner"), patch("streamlit.chat_message"):
            process_user_query("What is backpropagation?", pipeline_func=failing_pipeline)

        last_msg = st.session_state.messages[-1]
        assert last_msg["role"] == "assistant"
        assert last_msg["verdict"] == "FAIL"
        assert "API Error" in last_msg["content"]


# ============================================================================
# Scenario 13: Source Attribution Integrity
# ============================================================================

class TestScenario13SourceAttributionIntegrity:
    """Scenario 13: Chunks retain chunk_id, source, text, and ranking diagnostics."""

    def test_source_attribution_preserves_chunk_metadata_and_diagnostics(self):
        mock_chunks = [
            {
                "chunk_id": "transformer_summary_txt_chunk_0000",
                "source": "transformer_summary.txt",
                "text": "The Transformer is the first transduction model relying entirely on self-attention.",
                "dense_rank": 1,
                "bm25_rank": 2,
                "rrf_score": 0.032522,
                "metadata": {"matched_subqueries": ["What is Transformer?"]},
            }
        ]

        def mock_retriever(q: str, k: int = 4):
            return mock_chunks

        def mock_llm(messages, **kwargs):
            sys_msg = messages[0]["content"]
            if "strict, factual assistant" in sys_msg:
                return "The Transformer is a transduction model based entirely on self-attention."
            elif "strict fact-checker" in sys_msg:
                return "VERDICT: PASS\nREASON: Supported by transformer_summary.txt."
            return ""

        graph = build_rag_graph(retriever_func=mock_retriever, llm_func=mock_llm)
        final_state = run_rag_pipeline("What is Transformer?", graph=graph)

        chunk = final_state["chunks"][0]
        assert chunk["chunk_id"] == "transformer_summary_txt_chunk_0000"
        assert chunk["source"] == "transformer_summary.txt"
        assert chunk["dense_rank"] == 1
        assert chunk["bm25_rank"] == 2
        assert chunk["rrf_score"] == 0.032522
        assert "What is Transformer?" in chunk["metadata"]["matched_subqueries"]


# ============================================================================
# Scenario 14: Session History Integrity
# ============================================================================

class TestScenario14SessionHistoryIntegrity:
    """Scenario 14: Multi-turn session state maintains message order and isolates traces."""

    def test_multi_turn_session_preserves_chronology_and_independence(self):
        st.session_state.messages = []
        init_session_state()

        def mock_pipeline_turn1(q: str):
            return {
                "question": q,
                "original_question": q,
                "draft": "Answer to Turn 1.",
                "verdict": "PASS",
                "reason": "Verified turn 1.",
                "attempts": 0,
                "chunks": [{"chunk_id": "c1", "source": "doc1.txt", "text": "Turn 1 text"}],
                "trace": [{"node": "retrieve_node"}, {"node": "generate_node"}, {"node": "judge_node"}],
            }

        def mock_pipeline_turn2(q: str):
            return {
                "question": q,
                "original_question": q,
                "draft": "Answer to Turn 2.",
                "verdict": "PASS",
                "reason": "Verified turn 2.",
                "attempts": 0,
                "chunks": [{"chunk_id": "c2", "source": "doc2.txt", "text": "Turn 2 text"}],
                "trace": [{"node": "retrieve_node"}, {"node": "generate_node"}, {"node": "judge_node"}],
            }

        with patch("ui.chat.is_groq_configured", return_value=True), patch("streamlit.spinner"), patch("streamlit.chat_message"):
            process_user_query("Turn 1 Question", pipeline_func=mock_pipeline_turn1)
            process_user_query("Turn 2 Question", pipeline_func=mock_pipeline_turn2)

        messages = st.session_state.messages
        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "Turn 1 Question"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Answer to Turn 1."
        assert messages[2] == {"role": "user", "content": "Turn 2 Question"}
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "Answer to Turn 2."
        # Verify independence of chunk IDs
        assert messages[1]["chunks"][0]["chunk_id"] == "c1"
        assert messages[3]["chunks"][0]["chunk_id"] == "c2"


# ============================================================================
# Scenario 15: Document Upload and Indexing Integrity
# ============================================================================

class TestScenario15DocumentUploadAndIndexing:
    """Scenario 15: Ingestion validates .txt, chunks, detects duplicates, and syncs BM25."""

    def test_document_upload_deduplication_and_bm25_sync(self):
        mock_file = MagicMock()
        mock_file.name = "quantum_computing.txt"
        mock_file.read.return_value = (
            b"Quantum computing utilizes qubits for superposition and entanglement calculations. "
            b"Qubits can represent 0, 1, or both simultaneously through linear combinations."
        )

        mock_store = MagicMock()
        # First upload: indices 2 chunks
        mock_store.index_chunks.return_value = 2

        with patch("streamlit.sidebar.success"), patch("streamlit.sidebar.info"):
            with patch("src.hybrid_retriever.get_hybrid_retriever") as mock_get_hybrid:
                mock_hybrid = MagicMock()
                mock_get_hybrid.return_value = mock_hybrid

                handle_document_upload([mock_file], store=mock_store)

                assert mock_store.index_chunks.called
                assert mock_hybrid.sync_bm25.called

        # Second upload with duplicate detection (returns 0 new chunks)
        mock_store.index_chunks.return_value = 0
        with patch("streamlit.sidebar.info") as mock_info:
            handle_document_upload([mock_file], store=mock_store)
            mock_info.assert_called()


# ============================================================================
# Optional Live Groq Validation Suite
# ============================================================================

@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1" or not is_groq_configured(),
    reason="Live Groq tests require RUN_LIVE_TESTS=1 and GROQ_API_KEY environment variable",
)
class TestLiveGroqValidationOptional:
    """Optional live Groq validation test verifying live API compatibility."""

    def test_live_groq_answer_generation_and_judge(self):
        question = "How many encoder and decoder layers does the Transformer use?"
        final_state = run_rag_pipeline(question)
        assert final_state["verdict"] in ["PASS", "FAIL"]
        assert len(final_state["draft"]) > 0
        assert len(final_state["chunks"]) > 0
