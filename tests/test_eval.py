"""Automated unit tests for Alethia Evaluation & Benchmarking Harness (Phase 6).

Verifies:
1. Evaluation dataset integrity (30 questions across all 4 categories)
2. Retrieval evaluation calculations (Recall@K, MRR)
3. Hallucination detection classification (TP, TN, FP, FN, Precision, Recall, F1)
4. System metrics aggregation and abstention tracking
5. Deterministic mock execution with 0 external API credits
"""

from unittest.mock import MagicMock
import pytest

from eval.dataset import EVALUATION_DATASET, get_evaluation_dataset
from eval.retrieval_eval import evaluate_retrieval, is_chunk_relevant
from eval.hallucination_eval import evaluate_hallucination_detection
from eval.system_eval import evaluate_end_to_end_system


def test_evaluation_dataset_distribution_and_integrity():
    """Verify that dataset contains exactly 30 questions matching the required distribution."""
    dataset = get_evaluation_dataset()
    assert len(dataset) == 30

    categories = [d["category"] for d in dataset]
    assert categories.count("factual") == 10
    assert categories.count("out_of_scope") == 10
    assert categories.count("ambiguous") == 5
    assert categories.count("challenge") == 5

    # Check required fields
    for item in dataset:
        assert "id" in item
        assert "question" in item
        assert "expected_answer" in item
        assert "expected_behavior" in item
        if item["category"] == "out_of_scope":
            assert item["expected_behavior"] == "abstain"
            assert item["expected_answer"] == "Not in context."


def test_retrieval_evaluation_metrics_calculation():
    """Verify Recall@K and MRR formulas using a mock vector store."""
    mock_store = MagicMock()

    # Mock retrieve_chunks to return a match at Rank 1 for query 1, and Rank 2 for query 2
    def mock_retrieve_chunks(query, k=5):
        if "encoder and decoder" in query:
            return [
                {"chunk_id": "c1", "source": "transformer_summary.txt", "text": "six encoder layers and six decoder layers", "similarity": 0.9},
                {"chunk_id": "c2", "source": "other.txt", "text": "other", "similarity": 0.5},
            ]
        elif "BLEU score" in query:
            return [
                {"chunk_id": "c1", "source": "other.txt", "text": "other", "similarity": 0.8},
                {"chunk_id": "c2", "source": "transformer_summary.txt", "text": "28.4 BLEU for English-to-German translation", "similarity": 0.7},
            ]
        return [{"chunk_id": "c1", "source": "unrelated.txt", "text": "none", "similarity": 0.1}]

    mock_store.retrieve_chunks.side_effect = mock_retrieve_chunks

    mini_dataset = [
        EVALUATION_DATASET[0],  # transformer layers -> Rank 1
        EVALUATION_DATASET[1],  # BLEU score -> Rank 2
    ]

    results = evaluate_retrieval(store=mock_store, dataset=mini_dataset)

    assert results["total_queries_evaluated"] == 2
    # Q1 at Rank 1, Q2 at Rank 2:
    # Hits@1 = 1 / 2 = 0.5
    # Hits@3 = 2 / 2 = 1.0
    # MRR = (1/1 + 1/2) / 2 = (1.0 + 0.5) / 2 = 0.75
    assert results["recall_at_1"] == 0.5
    assert results["recall_at_3"] == 1.0
    assert results["hit_rate_at_1"] == 0.5
    assert results["hit_rate_at_3"] == 1.0
    assert results["mrr"] == 0.75


def test_hallucination_evaluation_confusion_matrix_math():
    """Verify TP, TN, FP, FN, Precision, Recall, and F1 calculations with mocked judge."""
    def mock_perfect_judge(messages, temperature=0.0, max_tokens=1024):
        user_prompt = messages[1]["content"]
        # Ground truth: if draft has hallucinated / contradictory claim, output FAIL, else PASS
        if "twelve encoder layers" in user_prompt or "TPU v4 chips" in user_prompt:
            return "VERDICT: FAIL\nREASON: Contradicts context."
        return "VERDICT: PASS\nREASON: Supported by context."

    test_pairs = [
        # 1 True Positive (Hallucination -> FAIL)
        {
            "id": "T1",
            "type": "contradictory",
            "question": "Q?",
            "context": "Context",
            "draft": "twelve encoder layers",
            "ground_truth_verdict": "FAIL",
            "is_hallucination": True,
        },
        # 1 True Negative (Supported -> PASS)
        {
            "id": "T2",
            "type": "supported",
            "question": "Q?",
            "context": "Context",
            "draft": "supported answer",
            "ground_truth_verdict": "PASS",
            "is_hallucination": False,
        },
    ]

    results = evaluate_hallucination_detection(llm_func=mock_perfect_judge, test_cases=test_pairs)

    assert results["total_cases"] == 2
    assert results["true_positives"] == 1
    assert results["true_negatives"] == 1
    assert results["false_positives"] == 0
    assert results["false_negatives"] == 0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1_score"] == 1.0
    assert results["accuracy"] == 1.0


def test_system_eval_abstention_and_metrics_accumulation():
    """Verify that evaluate_end_to_end_system correctly tracks abstentions and accuracies."""
    def mock_runner(query):
        if "Australia" in query:
            # Out-of-scope question properly abstains
            return {
                "question": query,
                "draft": "Not in context.",
                "verdict": "PASS",
                "reason": "Context lacks Australian capital.",
                "attempts": 0,
                "chunks": [],
                "trace": [{"node": "generate_node"}],
            }
        else:
            # Factual question answered correctly
            return {
                "question": query,
                "draft": "The architecture contains six encoder layers and six decoder layers.",
                "verdict": "PASS",
                "reason": "Supported by context.",
                "attempts": 0,
                "chunks": [{"chunk_id": "c1", "source": "t.txt"}],
                "trace": [{"node": "generate_node"}],
            }

    mini_dataset = [
        EVALUATION_DATASET[0],   # Factual question (FAC-01)
        EVALUATION_DATASET[10],  # Out-of-scope question (ABS-01, Australia)
    ]

    results = evaluate_end_to_end_system(dataset=mini_dataset, pipeline_func=mock_runner)

    assert results["total_evaluated"] == 2
    assert results["category_metrics"]["factual_accuracy"] == 0.1  # 1/10
    assert results["category_metrics"]["abstention_precision"] == 0.1  # 1/10
    assert results["cycle_statistics"]["average_cycles"] == 1.0
    assert results["cycle_statistics"]["maximum_retries"] == 0


def test_zero_rewrites_never_counts_as_self_correction():
    """Verify that an ambiguous query resolved with attempts=0 is counted as initial_success, NEVER self-correction."""
    ambiguous_item = [item for item in EVALUATION_DATASET if item["id"] == "AMB-01"]

    def mock_initial_success_runner(query):
        return {
            "question": query,
            "draft": "Six encoder layers and six decoder layers.",
            "verdict": "PASS",
            "reason": "Supported by context.",
            "attempts": 0,
            "chunks": [{"chunk_id": "c1", "source": "transformer_summary.txt"}],
            "trace": [{"node": "generate_node"}],
        }

    results = evaluate_end_to_end_system(dataset=ambiguous_item, pipeline_func=mock_initial_success_runner)
    detail = results["case_details"][0]
    summary = results["self_correction_summary"]

    # Must be classified as initial_success
    assert detail["classification"] == "initial_success"
    assert summary["initial_success_count"] == 1
    # Must NOT be counted as self-correction
    assert summary["successful_correction_count"] == 0
    assert summary["rewrites_triggered_count"] == 0
    assert summary["self_correction_success_rate"] == 0.0
    assert summary["first_attempt_success_rate"] == 100.0


def test_successful_rewrite_counted_correctly():
    """Verify that an ambiguous query resolved after confirmed cyclic trace is counted as successful_correction."""
    ambiguous_item = [item for item in EVALUATION_DATASET if item["id"] == "AMB-01"]

    def mock_successful_rewrite_runner(query):
        return {
            "question": "How many encoder and decoder layers does the base Transformer architecture use?",
            "draft": "Six encoder layers and six decoder layers.",
            "verdict": "PASS",
            "reason": "Supported by context after rewrite.",
            "attempts": 1,
            "chunks": [{"chunk_id": "c1", "source": "transformer_summary.txt"}],
            "trace": [
                {"node": "retrieve_node", "question": query},
                {"node": "generate_node", "draft": "Initial vague draft."},
                {"node": "judge_node", "verdict": "FAIL", "reason": "Initial draft was ungrounded."},
                {"node": "rewrite_node", "attempt": 1, "rewritten_question": "How many encoder layers?"},
                {"node": "retrieve_node", "question": "How many encoder layers?"},
                {"node": "generate_node", "draft": "Six encoder layers and six decoder layers."},
                {"node": "judge_node", "verdict": "PASS", "reason": "Correctly verified."},
            ],
        }

    results = evaluate_end_to_end_system(dataset=ambiguous_item, pipeline_func=mock_successful_rewrite_runner)
    detail = results["case_details"][0]
    summary = results["self_correction_summary"]

    assert detail["classification"] == "successful_correction"
    assert summary["initial_success_count"] == 0
    assert summary["rewrites_triggered_count"] == 1
    assert summary["successful_correction_count"] == 1
    assert summary["failed_correction_count"] == 0
    assert summary["self_correction_success_rate"] == 100.0


def test_attempts_without_initial_fail_verdict_rejected():
    """Verify that attempts > 0 without an initial FAIL verdict is not counted as successful_correction."""
    ambiguous_item = [item for item in EVALUATION_DATASET if item["id"] == "AMB-01"]

    def mock_bad_trace_runner(query):
        return {
            "question": query,
            "draft": "Six encoder layers and six decoder layers.",
            "verdict": "PASS",
            "reason": "Supported.",
            "attempts": 1,
            "chunks": [{"chunk_id": "c1", "source": "transformer_summary.txt"}],
            # Trace shows initial verdict was PASS, not FAIL
            "trace": [
                {"node": "retrieve_node"},
                {"node": "generate_node"},
                {"node": "judge_node", "verdict": "PASS"},
                {"node": "rewrite_node"},
                {"node": "retrieve_node"},
                {"node": "generate_node"},
                {"node": "judge_node", "verdict": "PASS"},
            ],
        }

    results = evaluate_end_to_end_system(dataset=ambiguous_item, pipeline_func=mock_bad_trace_runner)
    detail = results["case_details"][0]
    summary = results["self_correction_summary"]

    assert detail["classification"] == "failed_correction"
    assert summary["successful_correction_count"] == 0
    assert summary["failed_correction_count"] == 1


def test_attempts_without_subsequent_retrieval_rejected():
    """Verify that attempts > 0 without subsequent retrieval after rewrite is not counted as successful_correction."""
    ambiguous_item = [item for item in EVALUATION_DATASET if item["id"] == "AMB-01"]

    def mock_no_retrieval_runner(query):
        return {
            "question": query,
            "draft": "Six encoder layers and six decoder layers.",
            "verdict": "PASS",
            "reason": "Supported.",
            "attempts": 1,
            "chunks": [{"chunk_id": "c1", "source": "transformer_summary.txt"}],
            # Trace has no retrieve_node after rewrite_node
            "trace": [
                {"node": "retrieve_node"},
                {"node": "generate_node"},
                {"node": "judge_node", "verdict": "FAIL"},
                {"node": "rewrite_node"},
                {"node": "generate_node"},
                {"node": "judge_node", "verdict": "PASS"},
            ],
        }

    results = evaluate_end_to_end_system(dataset=ambiguous_item, pipeline_func=mock_no_retrieval_runner)
    detail = results["case_details"][0]
    summary = results["self_correction_summary"]

    assert detail["classification"] == "failed_correction"
    assert summary["successful_correction_count"] == 0
    assert summary["failed_correction_count"] == 1


def test_failed_rewrite_counted_correctly():
    """Verify that a query failing after max retries is classified as failed_correction."""
    ambiguous_item = [item for item in EVALUATION_DATASET if item["id"] == "AMB-01"]

    def mock_failed_rewrite_runner(query):
        return {
            "question": "Vague question still failing?",
            "draft": "Incorrect answer.",
            "verdict": "FAIL",
            "reason": "Unsupported by context.",
            "attempts": 2,
            "chunks": [{"chunk_id": "c1", "source": "transformer_summary.txt"}],
            "trace": [
                {"node": "generate_node"},
                {"node": "rewrite_node"},
                {"node": "generate_node"},
                {"node": "rewrite_node"},
                {"node": "generate_node"},
            ],
        }

    results = evaluate_end_to_end_system(dataset=ambiguous_item, pipeline_func=mock_failed_rewrite_runner)
    detail = results["case_details"][0]
    summary = results["self_correction_summary"]

    assert detail["classification"] == "failed_correction"
    assert summary["initial_success_count"] == 0
    assert summary["rewrites_triggered_count"] == 1
    assert summary["successful_correction_count"] == 0
    assert summary["failed_correction_count"] == 1
    assert summary["self_correction_success_rate"] == 0.0


def test_correct_abstention_counted_separately():
    """Verify that correct abstention on out-of-scope query is classified as abstention and counted separately."""
    out_of_scope_item = [item for item in EVALUATION_DATASET if item["id"] == "ABS-01"]

    def mock_abstain_runner(query):
        return {
            "question": query,
            "draft": "Not in context.",
            "verdict": "PASS",
            "reason": "Australia is not in the knowledge base.",
            "attempts": 0,
            "chunks": [],
            "trace": [{"node": "generate_node"}],
        }

    results = evaluate_end_to_end_system(dataset=out_of_scope_item, pipeline_func=mock_abstain_runner)
    detail = results["case_details"][0]

    assert detail["classification"] == "abstention"
    assert detail["success"] is True
    assert results["category_metrics"]["abstention_precision"] == 0.1


# =========================================================================
# Step 2: Retrieval Methodology Realignment & Ground Truth Unit Tests
# =========================================================================

def test_short_token_and_generic_word_false_positives_rejected():
    """Verify that short tokens ('1', '6', 'q', 'k') and generic words are rejected as relevance evidence."""
    # Query: FAC-01 (Transformer layers) with expected doc transformer_summary.txt
    gt = {
        "id": "FAC-01",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "required_keyphrases": ["six encoder layers and six decoder layers"],
        "expected_behavior": "answer",
    }

    # Chunk from same source containing short token '6' or digit '1' or common word 'decay' but not ground truth
    chunk_with_single_digit = {
        "chunk_id": "transformer_summary_txt_chunk_0000",
        "source": "transformer_summary.txt",
        "text": "Introduced in 2017 with 1 major breakthrough and 6 authors.",
    }
    is_rel, reason = is_chunk_relevant(chunk_with_single_digit, gt)
    assert is_rel is False
    assert reason == "no_match"

    # Chunk with generic technical term like 'model' or 'layers' alone
    chunk_with_generic_term = {
        "chunk_id": "transformer_summary_txt_chunk_0001",
        "source": "transformer_summary.txt",
        "text": "This model contains layers of neural operations.",
    }
    is_rel, reason = is_chunk_relevant(chunk_with_generic_term, gt)
    assert is_rel is False
    assert reason == "no_match"

    # Query: CHL-01 (AdamW) - ensure generic word 'decay' from chunk_0000 does not match
    chl_gt = {
        "id": "CHL-01",
        "relevant_source": "optimizers_guide.txt",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0003"],
        "required_keyphrases": ["decoupling weight decay from gradient updates"],
        "expected_behavior": "answer",
    }
    chunk_with_decay_word = {
        "chunk_id": "optimizers_guide_txt_chunk_0000",
        "source": "optimizers_guide.txt",
        "text": "Learning rate decay is a standard baseline in optimization.",
    }
    is_rel, reason = is_chunk_relevant(chunk_with_decay_word, chl_gt)
    assert is_rel is False
    assert reason == "no_match"


def test_correct_evidence_chunk_and_id_recognized():
    """Verify that chunks with matching explicit chunk ID or approved multi-token keyphrases are recognized."""
    gt = {
        "id": "FAC-01",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "required_keyphrases": ["six encoder layers and six decoder layers"],
        "expected_behavior": "answer",
    }

    # Match by chunk ID
    chunk_by_id = {
        "chunk_id": "transformer_summary_txt_chunk_0002",
        "source": "transformer_summary.txt",
        "text": "Different text but matching exact chunk ID.",
    }
    is_rel, reason = is_chunk_relevant(chunk_by_id, gt)
    assert is_rel is True
    assert "chunk_id:transformer_summary_txt_chunk_0002" in reason

    # Match by multi-token keyphrase
    chunk_by_phrase = {
        "chunk_id": "different_id",
        "source": "transformer_summary.txt",
        "text": "The base model contains six encoder layers and six decoder layers for translation.",
    }
    is_rel, reason = is_chunk_relevant(chunk_by_phrase, gt)
    assert is_rel is True
    assert "keyphrase:" in reason


def test_irrelevant_chunks_rejected():
    """Verify that chunks from different documents or unrelated topics are rejected."""
    gt = {
        "id": "FAC-01",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "required_keyphrases": ["six encoder layers and six decoder layers"],
        "expected_behavior": "answer",
    }

    # Different source document
    unrelated_chunk = {
        "chunk_id": "solar_system_txt_chunk_0001",
        "source": "solar_system.txt",
        "text": "Jupiter is the largest planet in the solar system.",
    }
    is_rel, reason = is_chunk_relevant(unrelated_chunk, gt)
    assert is_rel is False
    assert reason == "no_match"


def test_multiple_relevant_chunks_handled_correctly():
    """Verify that queries with multiple relevant chunks correctly identify the first encountered rank."""
    multi_gt = {
        "id": "FAC-05",
        "category": "factual",
        "question": "What is the key difference in penalty formulation between L1 Lasso and L2 Ridge regularization?",
        "relevant_source": "machine_learning_fundamentals.txt",
        "relevant_chunk_ids": [
            "machine_learning_fundamentals_txt_chunk_0003",
            "machine_learning_fundamentals_txt_chunk_0004",
        ],
        "required_keyphrases": [
            "absolute value of the model weights",
            "squared magnitude of weights",
        ],
        "expected_behavior": "answer",
    }

    mock_store = MagicMock()
    # Mock returns unrelated chunk at rank 1, chunk_0004 at rank 2, chunk_0003 at rank 3
    mock_store.retrieve_chunks.return_value = [
        {"chunk_id": "c_unrelated", "source": "unrelated.txt", "text": "foo", "similarity": 0.9},
        {"chunk_id": "machine_learning_fundamentals_txt_chunk_0004", "source": "machine_learning_fundamentals.txt", "text": "bar", "similarity": 0.8},
        {"chunk_id": "machine_learning_fundamentals_txt_chunk_0003", "source": "machine_learning_fundamentals.txt", "text": "baz", "similarity": 0.7},
    ]

    res = evaluate_retrieval(store=mock_store, dataset=[multi_gt])
    assert res["total_queries_evaluated"] == 1
    query_res = res["query_results"][0]
    assert query_res["rank_found"] == 2
    assert query_res["matched_chunk_id"] == "machine_learning_fundamentals_txt_chunk_0004"
    assert res["recall_at_1"] == 0.0
    assert res["recall_at_3"] == 1.0
    assert res["recall_at_5"] == 1.0
    assert res["mrr"] == 0.5


def test_unanswerable_queries_excluded_from_retrieval_recall():
    """Verify that unanswerable / out-of-scope queries are excluded from answerable recall metrics."""
    mixed_dataset = [
        {
            "id": "FAC-01",
            "category": "factual",
            "question": "Transformer layers?",
            "relevant_source": "transformer_summary.txt",
            "relevant_chunk_ids": ["c1"],
            "expected_behavior": "answer",
        },
        {
            "id": "ABS-01",
            "category": "out_of_scope",
            "question": "Capital of Australia?",
            "relevant_source": None,
            "relevant_chunk_ids": [],
            "expected_behavior": "abstain",
        },
    ]

    mock_store = MagicMock()
    mock_store.retrieve_chunks.return_value = [
        {"chunk_id": "c1", "source": "transformer_summary.txt", "text": "layers", "similarity": 0.9}
    ]

    res = evaluate_retrieval(store=mock_store, dataset=mixed_dataset)
    # Only 1 answerable query should be evaluated; ABS-01 excluded
    assert res["total_queries_evaluated"] == 1
    assert res["unanswerable_queries_excluded"] == 1
    assert res["recall_at_1"] == 1.0
    assert res["mrr"] == 1.0


def test_retrieval_metrics_mathematical_precision():
    """Verify Recall@1, Recall@3, Recall@5, and MRR mathematical calculations across 4 cases."""
    mock_store = MagicMock()

    # Create 4 queries where ranks found are:
    # Q1 -> Rank 1 (Hit@1, Hit@3, Hit@5, RR=1.0)
    # Q2 -> Rank 2 (Hit@3, Hit@5, RR=0.5)
    # Q3 -> Rank 4 (Hit@5, RR=0.25)
    # Q4 -> Miss (RR=0.0)
    cases = [
        {"id": "Q1", "category": "c", "question": "q1", "relevant_source": "s.txt", "relevant_chunk_ids": ["target_q1"], "expected_behavior": "answer"},
        {"id": "Q2", "category": "c", "question": "q2", "relevant_source": "s.txt", "relevant_chunk_ids": ["target_q2"], "expected_behavior": "answer"},
        {"id": "Q3", "category": "c", "question": "q3", "relevant_source": "s.txt", "relevant_chunk_ids": ["target_q3"], "expected_behavior": "answer"},
        {"id": "Q4", "category": "c", "question": "q4", "relevant_source": "s.txt", "relevant_chunk_ids": ["target_q4"], "expected_behavior": "answer"},
    ]

    def mock_retrieval(query, k=5):
        if query == "q1":
            return [{"chunk_id": "target_q1", "source": "s.txt", "similarity": 0.9}]
        elif query == "q2":
            return [
                {"chunk_id": "other1", "source": "s.txt", "similarity": 0.9},
                {"chunk_id": "target_q2", "source": "s.txt", "similarity": 0.8},
            ]
        elif query == "q3":
            return [
                {"chunk_id": "other1", "source": "s.txt", "similarity": 0.9},
                {"chunk_id": "other2", "source": "s.txt", "similarity": 0.8},
                {"chunk_id": "other3", "source": "s.txt", "similarity": 0.7},
                {"chunk_id": "target_q3", "source": "s.txt", "similarity": 0.6},
            ]
        else:
            return [{"chunk_id": "other1", "source": "s.txt", "similarity": 0.9}]

    mock_store.retrieve_chunks.side_effect = mock_retrieval

    res = evaluate_retrieval(store=mock_store, dataset=cases)
    assert res["total_queries_evaluated"] == 4
    # Recall@1: 1 / 4 = 0.25
    assert res["recall_at_1"] == 0.25
    # Recall@3: 2 / 4 = 0.50
    assert res["recall_at_3"] == 0.50
    # Recall@5: 3 / 4 = 0.75
    assert res["recall_at_5"] == 0.75
    # MRR: (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 0.4375
    assert res["mrr"] == 0.4375


def test_retrieval_evaluation_determinism():
    """Verify that repeated evaluation produces deterministic, reproducible metrics."""
    mock_store = MagicMock()
    mock_store.retrieve_chunks.return_value = [
        {"chunk_id": "transformer_summary_txt_chunk_0002", "source": "transformer_summary.txt", "similarity": 0.88}
    ]
    test_case = [EVALUATION_DATASET[0]]

    run1 = evaluate_retrieval(store=mock_store, dataset=test_case)
    run2 = evaluate_retrieval(store=mock_store, dataset=test_case)

    assert run1["recall_at_1"] == run2["recall_at_1"]
    assert run1["mrr"] == run2["mrr"]
    assert run1["query_results"] == run2["query_results"]


def test_recall_vs_hit_rate_distinction():
    """Verify that Recall@K and Hit Rate@K are mathematically distinct when multiple relevant chunks exist.

    Example required by audit:
    - Total relevant chunks: 4 (c1, c2, c3, c4).
    - Relevant chunks retrieved in top 3: 2 (c1, c2).
    - Expected Recall@3: 2 / 4 = 50.0%.
    - Expected Hit Rate@3: 1 / 1 = 100.0%.
    """
    mock_store = MagicMock()
    # Mock returns 2 relevant chunks and 1 irrelevant chunk in top 3
    mock_store.retrieve_chunks.return_value = [
        {"chunk_id": "c1", "source": "doc.txt", "text": "target content 1", "similarity": 0.95},
        {"chunk_id": "unrelated", "source": "unrelated.txt", "text": "unrelated", "similarity": 0.85},
        {"chunk_id": "c2", "source": "doc.txt", "text": "target content 2", "similarity": 0.75},
        {"chunk_id": "c3", "source": "doc.txt", "text": "target content 3", "similarity": 0.65},
        {"chunk_id": "c4", "source": "doc.txt", "text": "target content 4", "similarity": 0.55},
    ]

    single_case = [
        {
            "id": "MULTI-01",
            "category": "factual",
            "question": "Multi-chunk query?",
            "relevant_source": "doc.txt",
            "relevant_chunk_ids": ["c1", "c2", "c3", "c4"],
            "expected_behavior": "answer",
        }
    ]

    res = evaluate_retrieval(store=mock_store, dataset=single_case)

    # In Top 3:
    # 2 out of 4 relevant chunks retrieved -> Recall@3 = 0.50 (50.0%)
    assert res["recall_at_3"] == 0.50
    # At least one relevant chunk retrieved in Top 3 -> Hit Rate@3 = 1.0 (100.0%)
    assert res["hit_rate_at_3"] == 1.0

    # In Top 1:
    # 1 out of 4 relevant chunks retrieved -> Recall@1 = 0.25 (25.0%)
    assert res["recall_at_1"] == 0.25
    # At least one relevant chunk in Top 1 -> Hit Rate@1 = 1.0 (100.0%)
    assert res["hit_rate_at_1"] == 1.0

    # In Top 5:
    # 4 out of 4 relevant chunks retrieved -> Recall@5 = 1.0 (100.0%)
    assert res["recall_at_5"] == 1.0
    assert res["hit_rate_at_5"] == 1.0




