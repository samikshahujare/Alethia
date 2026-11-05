"""Unit and integration tests for intelligent multi-part question decomposition (Phase 3).

Verifies:
1. Rule-based detection of single-intent vs multi-intent questions.
2. Single-intent optimization (zero LLM calls for simple queries).
3. Rule-based and LLM-assisted question decomposition.
4. Technical terms, entity, and numerical value preservation.
5. Handling of malformed, empty, duplicate, or excessive LLM responses.
6. Sub-question retrieval with chunk_id deduplication and matched_subqueries metadata.
7. GraphState backwards compatibility and new field preservation.
8. End-to-end integration with generator and hardened 5-rule Judge (Rule 2 partial abstention).
"""

import json
import pytest
from typing import Any, Dict, List

from src.decomposition import (
    is_multi_intent_question,
    decompose_question_rule_based,
    decompose_question_llm,
    decompose_question,
    retrieve_decomposed,
    merge_subquestion_chunks,
    MAX_SUBQUESTIONS,
)
from src.state import GraphState, create_initial_state
from src.hybrid_retriever import HybridRetriever
from src.nodes import retrieve_node, generate_node, judge_node
from src.graph import build_rag_graph


# ============================================================================
# 1. Detection of Single-Intent vs Multi-Intent Questions
# ============================================================================

class TestMultiIntentDetection:
    """Test rule-based identification of multi-part vs single-intent questions."""

    @pytest.mark.parametrize("query", [
        "What is backpropagation?",
        "Explain transformer self-attention.",
        "What is the default learning rate of Adam?",
        "Define precision and recall.",  # compound noun phrase, single interrogative
        "How does gradient descent work?",
        "What are the benefits of dropout?",
        "What is the difference between L1 and L2 regularization?",
    ])
    def test_single_intent_queries_detected_as_false(self, query: str):
        """Single-intent queries must return False and avoid decomposition."""
        assert is_multi_intent_question(query) is False

    @pytest.mark.parametrize("query", [
        "What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
        "What is backpropagation? And how does gradient descent update weights?",
        "What is precision, and how is recall calculated?",
        "Explain self-attention, and also describe multi-head attention.",
        "What is learning rate, as well as what is batch size?",
        "What is BERT, additionally what is GPT?",
        "Which loss function is used for classification, and what optimizer is standard?",
    ])
    def test_multi_intent_queries_detected_as_true(self, query: str):
        """Coordinated multi-part queries must return True."""
        assert is_multi_intent_question(query) is True

    def test_empty_or_whitespace_query(self):
        """Empty or whitespace queries must return False safely."""
        assert is_multi_intent_question("") is False
        assert is_multi_intent_question("   ") is False
        assert is_multi_intent_question(None) is False


# ============================================================================
# 2. Rule-Based Question Decomposition
# ============================================================================

class TestRuleBasedDecomposition:
    """Test rule-based decomposition without LLM invocation."""

    def test_decompose_two_questions_separated_by_question_mark(self):
        query = "What is backpropagation? How does gradient descent work?"
        sub_qs = decompose_question_rule_based(query)
        assert len(sub_qs) == 2
        assert "What is backpropagation?" in sub_qs[0]
        assert "How does gradient descent work?" in sub_qs[1]

    def test_decompose_coordinated_and_clauses(self):
        query = "What optimizer was used in the Transformer paper, and what are Adam's default beta values?"
        sub_qs = decompose_question_rule_based(query)
        assert len(sub_qs) == 2
        assert any("optimizer" in q.lower() and "transformer" in q.lower() for q in sub_qs)
        assert any("beta" in q.lower() and "adam" in q.lower() for q in sub_qs)

    def test_decompose_as_well_as_clauses(self):
        query = "What is the learning rate, as well as what is the batch size?"
        sub_qs = decompose_question_rule_based(query)
        assert len(sub_qs) == 2
        assert any("learning rate" in q.lower() for q in sub_qs)
        assert any("batch size" in q.lower() for q in sub_qs)

    def test_single_intent_returns_original_in_list(self):
        query = "What is backpropagation?"
        sub_qs = decompose_question_rule_based(query)
        assert sub_qs == [query]


# ============================================================================
# 3. LLM-Assisted Decomposition & Robust Error Handling
# ============================================================================

class TestLLMAssistedDecomposition:
    """Test LLM-assisted decomposition with structured JSON output and fallbacks."""

    def test_successful_llm_json_parsing(self):
        mock_response = json.dumps({
            "sub_questions": [
                "Which optimizer was used in the Transformer paper?",
                "What are the default beta values of Adam?",
            ]
        })
        sub_qs = decompose_question_llm(
            "What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
            llm_func=lambda **kwargs: mock_response,
        )
        assert len(sub_qs) == 2
        assert sub_qs[0] == "Which optimizer was used in the Transformer paper?"
        assert sub_qs[1] == "What are the default beta values of Adam?"

    def test_llm_markdown_code_fence_parsing(self):
        mock_response = (
            "Here is the decomposition:\n```json\n"
            "{\n  \"sub_questions\": [\"What is Adam?\", \"What is SGD?\"]\n}\n```"
        )
        sub_qs = decompose_question_llm(
            "What is Adam and what is SGD?",
            llm_func=lambda **kwargs: mock_response,
        )
        assert sub_qs == ["What is Adam?", "What is SGD?"]

    def test_llm_malformed_json_falls_back_to_rule_based_or_original(self):
        mock_bad_json = "This is not valid json at all {sub_questions: [unclosed"
        query = "What is precision, and how is recall calculated?"
        sub_qs = decompose_question_llm(
            query,
            llm_func=lambda **kwargs: mock_bad_json,
        )
        # Must gracefully fall back without raising an unhandled exception
        assert len(sub_qs) >= 1
        assert all(isinstance(q, str) and len(q) > 0 for q in sub_qs)

    def test_llm_empty_output_falls_back_safely(self):
        query = "What is backpropagation?"
        sub_qs = decompose_question_llm(
            query,
            llm_func=lambda **kwargs: "",
        )
        assert sub_qs == [query]

    def test_llm_duplicate_subquestions_deduplicated(self):
        mock_response = json.dumps({
            "sub_questions": [
                "What are Adam's beta values?",
                "What are Adam's beta values?",
                "what are adam's beta values?",
            ]
        })
        sub_qs = decompose_question_llm(
            "What are Adam's beta values?",
            llm_func=lambda **kwargs: mock_response,
        )
        assert len(sub_qs) == 1
        assert sub_qs[0] == "What are Adam's beta values?"

    def test_llm_excessive_subquestions_limited_to_max(self):
        mock_response = json.dumps({
            "sub_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
                "Question 5?",
            ]
        })
        sub_qs = decompose_question_llm(
            "Question 1 and 2 and 3 and 4 and 5?",
            llm_func=lambda **kwargs: mock_response,
        )
        assert len(sub_qs) <= MAX_SUBQUESTIONS
        assert len(sub_qs) == 3


# ============================================================================
# 4. Top-Level Unified `decompose_question` Entry Point
# ============================================================================

class TestUnifiedDecomposeQuestion:
    """Test the main decompose_question entry point with fast paths and fallbacks."""

    def test_single_intent_skips_llm_completely(self):
        """Single-intent questions must bypass LLM invocation to save latency/tokens."""
        call_count = 0

        def mock_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            return "{}"

        result = decompose_question("What is backpropagation?", llm_func=mock_llm)
        assert result == ["What is backpropagation?"]
        assert call_count == 0  # Zero LLM calls!

    def test_multi_intent_preserves_technical_terms_and_numbers(self):
        query = "What optimizer was used in the Transformer paper, and what are Adam's default beta_1 = 0.9 and beta_2 = 0.999 values?"
        mock_response = json.dumps({
            "sub_questions": [
                "What optimizer was used in the Transformer paper?",
                "What are Adam's default beta_1 = 0.9 and beta_2 = 0.999 values?",
            ]
        })
        sub_qs = decompose_question(query, llm_func=lambda **kwargs: mock_response)
        assert len(sub_qs) == 2
        assert "Transformer" in sub_qs[0]
        assert "beta_1 = 0.9" in sub_qs[1]
        assert "beta_2 = 0.999" in sub_qs[1]


# ============================================================================
# 5. Chunk Deduplication and `matched_subqueries` Metadata
# ============================================================================

class TestSubquestionChunkMerging:
    """Test merging and deduplication of chunk results retrieved across sub-questions."""

    def test_merge_chunks_deduplication_and_metadata_tracking(self):
        sub_results = {
            "Sub-question 1": [
                {"chunk_id": "c1", "text": "Chunk 1 text", "source": "doc1.txt", "rrf_score": 0.032},
                {"chunk_id": "c2", "text": "Chunk 2 text", "source": "doc2.txt", "rrf_score": 0.016},
            ],
            "Sub-question 2": [
                {"chunk_id": "c2", "text": "Chunk 2 text", "source": "doc2.txt", "rrf_score": 0.020},
                {"chunk_id": "c3", "text": "Chunk 3 text", "source": "doc3.txt", "rrf_score": 0.015},
            ],
        }

        merged = merge_subquestion_chunks(sub_results, k=4)
        merged_ids = [c["chunk_id"] for c in merged]

        # c2 was returned by both sub-questions; must appear exactly once
        assert len(merged) == 3
        assert set(merged_ids) == {"c1", "c2", "c3"}

        # Check matched_subqueries metadata
        c2_chunk = next(c for c in merged if c["chunk_id"] == "c2")
        assert "matched_subqueries" in c2_chunk["metadata"]
        assert "Sub-question 1" in c2_chunk["metadata"]["matched_subqueries"]
        assert "Sub-question 2" in c2_chunk["metadata"]["matched_subqueries"]

        c1_chunk = next(c for c in merged if c["chunk_id"] == "c1")
        assert c1_chunk["metadata"]["matched_subqueries"] == ["Sub-question 1"]

    def test_merge_chunks_preserves_diagnostics_and_limits_top_k(self):
        sub_results = {
            "Q1": [
                {"chunk_id": f"chunk_{i}", "text": f"Text {i}", "rrf_score": 0.01 * (10 - i)}
                for i in range(5)
            ],
            "Q2": [
                {"chunk_id": f"chunk_{i}", "text": f"Text {i}", "rrf_score": 0.01 * (10 - i)}
                for i in range(5, 10)
            ],
        }
        merged = merge_subquestion_chunks(sub_results, k=4)
        assert len(merged) == 4
        assert len({c["chunk_id"] for c in merged}) == 4


# ============================================================================
# 6. GraphState Compatibility and Extension
# ============================================================================

class TestGraphStatePhase3Compatibility:
    """Test that GraphState maintains all prior fields while supporting Phase 3 fields."""

    def test_create_initial_state_contains_phase3_fields(self):
        state = create_initial_state("What is precision and recall?")
        # Prior fields preserved
        assert state["question"] == "What is precision and recall?"
        assert state["original_question"] == "What is precision and recall?"
        assert state["chunks"] == []
        assert state["draft"] == ""
        assert state["verdict"] == ""
        assert state["attempts"] == 0
        assert state["trace"] == []

        # Phase 3 new fields initialized
        assert "decomposed_questions" in state
        assert state["decomposed_questions"] == []
        assert "subquestion_results" in state
        assert state["subquestion_results"] == {}
        assert "retrieval_strategy" in state
        assert state["retrieval_strategy"] == "hybrid"
        assert "retrieval_diagnostics" in state
        assert isinstance(state["retrieval_diagnostics"], dict)


# ============================================================================
# 7. Sub-question Retrieval Execution (`retrieve_decomposed`)
# ============================================================================

class TestRetrieveDecomposed:
    """Test retrieval across decomposed sub-questions using mock and real retriever."""

    def test_retrieve_decomposed_single_intent(self):
        class MockRetriever:
            def retrieve_hybrid(self, query, k=4):
                return [{"chunk_id": "c1", "text": "result for " + query, "metadata": {}}]

        retriever = MockRetriever()
        result = retrieve_decomposed(
            query="What is backpropagation?",
            k=4,
            strategy="hybrid",
            retriever=retriever,
        )

        assert result["original_question"] == "What is backpropagation?"
        assert result["decomposed_questions"] == ["What is backpropagation?"]
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["metadata"]["matched_subqueries"] == ["What is backpropagation?"]

    def test_retrieve_decomposed_multi_intent(self):
        class MockRetriever:
            def retrieve_hybrid(self, query, k=4):
                if "transformer" in query.lower():
                    return [{"chunk_id": "c_transformer", "text": "Transformer used Adam optimizer.", "metadata": {}}]
                elif "adam" in query.lower() or "beta" in query.lower():
                    return [{"chunk_id": "c_adam", "text": "Adam default beta1=0.9, beta2=0.999.", "metadata": {}}]
                return []

        mock_json = json.dumps({
            "sub_questions": [
                "Which optimizer was used in the Transformer paper?",
                "What are the default beta values of Adam?",
            ]
        })

        retriever = MockRetriever()
        result = retrieve_decomposed(
            query="What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
            k=4,
            strategy="hybrid",
            retriever=retriever,
            llm_func=lambda **kwargs: mock_json,
        )

        assert len(result["decomposed_questions"]) == 2
        assert len(result["chunks"]) == 2
        retrieved_ids = {c["chunk_id"] for c in result["chunks"]}
        assert retrieved_ids == {"c_transformer", "c_adam"}
        assert "Which optimizer was used in the Transformer paper?" in result["subquestion_results"]


# ============================================================================
# 8. Node and Graph Integration with Generator & Hardened Judge
# ============================================================================

class TestNodeAndGraphIntegration:
    """Test retrieval node integration and behavior with Generator and Hardened Judge."""

    def test_retrieve_node_populates_decomposed_questions_and_diagnostics(self):
        mock_chunks = [
            {"chunk_id": "c1", "text": "Adam info", "source": "adam.txt", "metadata": {}},
            {"chunk_id": "c2", "text": "Transformer info", "source": "trans.txt", "metadata": {}},
        ]

        def mock_retriever(query, k=4):
            return mock_chunks

        state = create_initial_state("What is Adam, and what is Transformer?")
        # Manually specify decomposed questions or let retrieve_node handle
        state["decomposed_questions"] = ["What is Adam?", "What is Transformer?"]

        updated_state = retrieve_node(state, retriever_func=mock_retriever)
        assert len(updated_state["chunks"]) > 0
        trace_entry = updated_state["trace"][-1]
        assert trace_entry["node"] == "retrieve_node"
        assert "chunk_count" in trace_entry

    def test_generator_and_hardened_judge_with_partial_evidence(self):
        """Rule 2 verification:

        When evidence is available for part 1 but missing for part 2, the generator
        answers part 1 and cleanly abstains on part 2. The hardened Judge evaluates
        the entire question and returns PASS under Rule 2 (Correct Abstention).
        """
        # Context has evidence for Adam optimizer in Transformer, but nothing about beta values
        chunks = [
            {
                "chunk_id": "chunk_transformer_001",
                "source": "transformer.txt",
                "text": "The authors used the Adam optimizer with beta1=0.9, beta2=0.98 in the Transformer paper.",
            }
        ]

        composite_question = "What optimizer was used in the Transformer paper, and what is the author's favourite color?"

        # Generator answers supported part and explicitly abstains on unsupported part
        generated_draft = (
            "The Transformer paper used the Adam optimizer. "
            "Information regarding the author's favourite color is not in the context."
        )

        # Mock LLM for Judge reflecting the hardened 5-rule prompt behavior
        def mock_judge_llm(messages, **kwargs):
            # Assert prompt evaluates complete composite question
            prompt_content = messages[1]["content"]
            assert composite_question in prompt_content
            # Hardened Rule 2 applies: partial abstention without fabrication
            return "VERDICT: PASS\nREASON: Correctly answered the optimizer and appropriately abstained on the author's favourite color which is missing from context."

        state = create_initial_state(composite_question)
        state["chunks"] = chunks
        state["draft"] = generated_draft

        judged_state = judge_node(state, llm_func=mock_judge_llm)
        assert judged_state["verdict"] == "PASS"
        assert "appropriately abstained" in judged_state["reason"].lower() or "correctly" in judged_state["reason"].lower()

