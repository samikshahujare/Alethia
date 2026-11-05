"""Automated unit tests for Alethia Real Self-Correction Evaluation Harness (Step 3).

Verifies:
1. Dataset schema, taxonomy, and integrity (16 cases, answerability, expected_decision).
2. Strict 10-step chronological trace verification logic (all pass, individual step failures).
3. Query-match verification in subsequent retrieval.
4. Ground-truth chunk ID verification in post-rewrite retrieval (different chunks alone do not suffice).
5. Case-specific mandatory ground-truth facts rubric (substring heuristic limitations documented).
6. Correct boundary query classification (abstention, never initial_success).
7. Proper handling of zero correction-required denominator ('N/A').
8. Proper handling of zero rewrites for max rewrite compliance ('N/A').
9. Overall abstention rate calculated against total evaluated suite (4/16 = 25.0%).
10. Preservation of 12/12 answer accuracy and 16/16 decision correctness.
11. Rejection of abstention on genuinely answerable queries.
12. Mutually exclusive classification logic.
13. Attempt history extraction across multi-cycle traces.
14. Benchmark metric calculation mathematics and edge cases.
"""

from typing import Any, Dict, List
import pytest

from eval.self_correction_eval import (
    SELF_CORRECTION_DATASET,
    classify_self_correction_case,
    evaluate_self_correction_benchmark,
    extract_attempt_history,
    verify_strict_self_correction_trace,
)


def _build_mock_successful_trace() -> List[Dict[str, Any]]:
    """Helper to construct a valid 10-step successful self-correction execution trace."""
    return [
        {
            "node": "retrieve_node",
            "question": "What is the penalty that creates sparse features?",
            "chunk_count": 4,
            "chunk_ids": ["c_gen_01", "c_gen_02", "c_gen_03", "c_gen_04"],
            "sources": ["machine_learning_fundamentals.txt"],
        },
        {
            "node": "generate_node",
            "question": "What is the penalty that creates sparse features?",
            "chunk_count": 4,
            "draft": "Regularization penalizes weights to prevent overfitting.",
        },
        {
            "node": "judge_node",
            "question": "What is the penalty that creates sparse features?",
            "draft": "Regularization penalizes weights to prevent overfitting.",
            "verdict": "FAIL",
            "reason": "Does not identify which specific penalty creates sparse features.",
        },
        {
            "node": "rewrite_node",
            "attempt": 1,
            "original_question": "What is the penalty that creates sparse features?",
            "previous_question": "What is the penalty that creates sparse features?",
            "rewritten_question": "L1 Lasso regularization penalty for sparse feature representations",
            "verification_reason": "Does not identify which specific penalty creates sparse features.",
        },
        {
            "node": "retrieve_node",
            "question": "L1 Lasso regularization penalty for sparse feature representations",
            "chunk_count": 4,
            "chunk_ids": ["machine_learning_fundamentals_txt_chunk_0003", "c_gen_02", "c_gen_03", "c_gen_04"],
            "sources": ["machine_learning_fundamentals.txt"],
        },
        {
            "node": "generate_node",
            "question": "L1 Lasso regularization penalty for sparse feature representations",
            "chunk_count": 4,
            "draft": "L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        },
        {
            "node": "judge_node",
            "question": "L1 Lasso regularization penalty for sparse feature representations",
            "draft": "L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
            "verdict": "PASS",
            "reason": "Directly supported by machine_learning_fundamentals_txt_chunk_0003.",
        },
        {
            "event": "workflow_completed",
            "final_verdict": "PASS",
            "total_attempts": 1,
            "total_trace_steps": 8,
        },
    ]


def test_self_correction_dataset_integrity():
    """Verify dataset taxonomy, answerability, and expected_decision integrity."""
    assert len(SELF_CORRECTION_DATASET) == 16
    case_ids = [c["id"] for c in SELF_CORRECTION_DATASET]
    assert len(case_ids) == len(set(case_ids)), "Duplicate case IDs detected!"

    required_keys = {
        "id",
        "question",
        "expected_answer",
        "relevant_chunk_ids",
        "mandatory_facts",
        "expected_decision",
        "answerability",
        "correction_reason",
        "category",
    }

    answerable_count = 0
    out_of_scope_count = 0
    corpus_boundary_count = 0

    for case in SELF_CORRECTION_DATASET:
        assert required_keys.issubset(case.keys()), f"Missing keys in {case['id']}"
        assert case["id"].startswith("SC-")
        assert len(case["question"].strip()) > 0
        assert len(case["expected_answer"].strip()) > 0
        assert len(case["correction_reason"].strip()) > 0

        assert case["answerability"] in {"answerable", "unanswerable"}
        assert case["expected_decision"] in {"answer", "abstain"}

        if case["answerability"] == "answerable":
            answerable_count += 1
            assert case["expected_decision"] == "answer"
            assert len(case["mandatory_facts"]) > 0
            assert len(case["relevant_chunk_ids"]) > 0
        else:
            assert case["expected_decision"] == "abstain"
            assert case["expected_answer"] == "Not in context."
            assert len(case["relevant_chunk_ids"]) == 0
            assert len(case["mandatory_facts"]) == 0
            if case["category"] == "corpus_boundary":
                corpus_boundary_count += 1
            else:
                out_of_scope_count += 1

    assert answerable_count == 12, "Must contain exactly 12 answerable cases (SC-01..11, SC-16)"
    assert out_of_scope_count == 2, "Must contain exactly 2 out-of-scope cases (SC-12, SC-13)"
    assert corpus_boundary_count == 2, "Must contain exactly 2 corpus-boundary cases (SC-14, SC-15)"


def test_strict_self_correction_trace_all_10_steps_passing():
    """Verify that a compliant trace passes all 10 chronological verification criteria."""
    trace = _build_mock_successful_trace()
    orig_q = "What is the penalty that creates sparse features?"
    final_v = "PASS"
    final_draft = (
        "L1 regularization, also known as Lasso regression, introduces a penalty "
        "proportional to the absolute values of the weights, producing sparse feature representations."
    )
    mand_facts = ["l1", "regularization", "sparse"]
    rel_chunks = ["machine_learning_fundamentals_txt_chunk_0003"]

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question=orig_q,
        final_verdict=final_v,
        final_draft=final_draft,
        mandatory_facts=mand_facts,
        relevant_chunk_ids=rel_chunks,
    )

    assert is_verified is True
    assert all(criteria.values()) is True
    assert criteria["step_01_initial_retrieval"] is True
    assert criteria["step_02_initial_generate"] is True
    assert criteria["step_03_initial_judge_fail"] is True
    assert criteria["step_04_rewrite_executed"] is True
    assert criteria["step_05_query_changed"] is True
    assert criteria["step_06_subsequent_retrieval"] is True
    assert criteria["step_06_subsequent_retrieval_uses_rewrite"] is True
    assert criteria["step_07_new_chunks_retrieved"] is True
    assert criteria["step_07_ground_truth_chunks_retrieved"] is True
    assert criteria["step_08_revised_answer_draft"] is True
    assert criteria["step_09_final_judge_pass"] is True
    assert criteria["step_09_final_judge_pass_after_generate"] is True
    assert criteria["step_10_final_answer_grounded"] is True
    assert criteria["step_10_all_mandatory_facts_present"] is True


@pytest.mark.parametrize(
    "step_to_break, expected_failed_step",
    [
        ("no_init_retrieval", "step_01_initial_retrieval"),
        ("no_init_generate", "step_02_initial_generate"),
        ("init_judge_pass", "step_03_initial_judge_fail"),
        ("no_rewrite_node", "step_04_rewrite_executed"),
        ("query_unchanged", "step_05_query_changed"),
        ("no_subsequent_retrieval", "step_06_subsequent_retrieval"),
        ("wrong_subsequent_query", "step_06_subsequent_retrieval_uses_rewrite"),
        ("identical_chunks", "step_07_new_chunks_retrieved"),
        ("unrelated_chunks_missing_ground_truth", "step_07_ground_truth_chunks_retrieved"),
        ("no_revised_draft", "step_08_revised_answer_draft"),
        ("final_judge_fail", "step_09_final_judge_pass"),
        ("judge_before_revised_generate", "step_09_final_judge_pass_after_generate"),
        ("missing_mandatory_facts", "step_10_all_mandatory_facts_present"),
    ],
)
def test_strict_self_correction_rejects_missing_any_step(step_to_break: str, expected_failed_step: str):
    """Verify that failing ANY of the 10 chronological criteria strictly rejects successful_correction."""
    trace = _build_mock_successful_trace()
    orig_q = "What is the penalty that creates sparse features?"
    final_v = "PASS"
    final_draft = (
        "L1 regularization, also known as Lasso regression, introduces a penalty "
        "proportional to the absolute values of the weights, producing sparse feature representations."
    )
    mand_facts = ["l1", "regularization", "sparse"]
    rel_chunks = ["machine_learning_fundamentals_txt_chunk_0003"]

    if step_to_break == "no_init_retrieval":
        trace = [e for e in trace if e.get("node") != "retrieve_node"]
    elif step_to_break == "no_init_generate":
        trace = [e for e in trace if e.get("node") != "generate_node"]
    elif step_to_break == "init_judge_pass":
        trace[2]["verdict"] = "PASS"
    elif step_to_break == "no_rewrite_node":
        trace = [e for e in trace if e.get("node") != "rewrite_node"]
    elif step_to_break == "query_unchanged":
        trace[3]["rewritten_question"] = orig_q
    elif step_to_break == "no_subsequent_retrieval":
        trace = trace[:4] + trace[5:]  # Remove second retrieve
    elif step_to_break == "wrong_subsequent_query":
        trace[4]["question"] = orig_q  # Retrieval didn't use rewritten query
    elif step_to_break == "identical_chunks":
        trace[4]["chunk_ids"] = list(trace[0]["chunk_ids"])  # Exactly same chunks
    elif step_to_break == "unrelated_chunks_missing_ground_truth":
        trace[4]["chunk_ids"] = ["unrelated_chunk_001", "unrelated_chunk_002"]  # New chunks, but not ground truth
    elif step_to_break == "no_revised_draft":
        trace[5]["draft"] = ""
    elif step_to_break == "final_judge_fail":
        trace[6]["verdict"] = "FAIL"
        final_v = "FAIL"
    elif step_to_break == "judge_before_revised_generate":
        # Swap judge and revised generate so judge occurs BEFORE generate
        trace[5], trace[6] = trace[6], trace[5]
    elif step_to_break == "missing_mandatory_facts":
        final_draft = "L1 penalty creates sparsity."  # Missing "regularization"

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question=orig_q,
        final_verdict=final_v,
        final_draft=final_draft,
        mandatory_facts=mand_facts,
        relevant_chunk_ids=rel_chunks,
    )

    assert is_verified is False, f"Expected verification failure when {step_to_break}"
    assert criteria[expected_failed_step] is False, f"Expected {expected_failed_step} to be False"


def test_different_chunks_alone_do_not_establish_useful_evidence():
    """Verify that merely retrieving different chunks without relevant ground-truth chunk IDs is strictly rejected."""
    trace = _build_mock_successful_trace()
    # Different chunks, but target_chunks is empty
    trace[4]["chunk_ids"] = ["unverified_chunk_alpha", "unverified_chunk_beta"]

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question="What is the penalty that creates sparse features?",
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=["l1", "regularization"],
        relevant_chunk_ids=[],  # No verified ground-truth chunk IDs provided
    )

    assert is_verified is False
    assert criteria["step_07_ground_truth_chunks_retrieved"] is False, (
        "Different chunks without verified ground-truth chunk IDs must NOT be accepted as useful evidence!"
    )


def test_newly_retrieved_evidence_verification_scenarios():
    """Verify all 4 evidence verification scenarios required by final Step 3 audit:
    1. A relevant chunk is retrieved for the first time after rewriting: PASS.
    2. A relevant chunk appears in both initial and post-rewrite results: FAIL.
    3. Post-rewrite retrieval returns only unrelated new chunks: FAIL.
    4. No ground-truth chunk IDs are supplied: FAIL.
    """
    target_chunk = "machine_learning_fundamentals_txt_chunk_0003"
    mand_facts = ["l1", "regularization"]
    orig_q = "What is the penalty that creates sparse features?"

    # Scenario 1: Relevant chunk retrieved for the first time after rewrite -> PASS
    trace_first_time = _build_mock_successful_trace()
    is_verified_1, criteria_1 = verify_strict_self_correction_trace(
        trace=trace_first_time,
        original_question=orig_q,
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=mand_facts,
        relevant_chunk_ids=[target_chunk],
    )
    assert is_verified_1 is True
    assert criteria_1["step_07_ground_truth_chunks_retrieved"] is True

    # Scenario 2: Relevant chunk appears in both initial and post-rewrite -> FAIL as evidence of newly retrieved information
    trace_both = _build_mock_successful_trace()
    trace_both[0]["chunk_ids"] = [target_chunk, "c_gen_01", "c_gen_02"]
    trace_both[4]["chunk_ids"] = [target_chunk, "c_new_01", "c_new_02"]
    is_verified_2, criteria_2 = verify_strict_self_correction_trace(
        trace=trace_both,
        original_question=orig_q,
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=mand_facts,
        relevant_chunk_ids=[target_chunk],
    )
    assert is_verified_2 is False
    assert criteria_2["step_07_ground_truth_chunks_retrieved"] is False, (
        "Ground-truth chunk present in both initial and post-rewrite must NOT count as newly retrieved evidence!"
    )

    # Scenario 3: Post-rewrite returns only unrelated new chunks -> FAIL
    trace_unrelated = _build_mock_successful_trace()
    trace_unrelated[0]["chunk_ids"] = ["c_init_01", "c_init_02"]
    trace_unrelated[4]["chunk_ids"] = ["c_unrelated_01", "c_unrelated_02"]
    is_verified_3, criteria_3 = verify_strict_self_correction_trace(
        trace=trace_unrelated,
        original_question=orig_q,
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=mand_facts,
        relevant_chunk_ids=[target_chunk],
    )
    assert is_verified_3 is False
    assert criteria_3["step_07_ground_truth_chunks_retrieved"] is False, (
        "Unrelated new chunks must fail ground-truth evidence verification!"
    )

    # Scenario 4: No ground-truth chunk IDs supplied -> FAIL
    trace_no_gt = _build_mock_successful_trace()
    is_verified_4, criteria_4 = verify_strict_self_correction_trace(
        trace=trace_no_gt,
        original_question=orig_q,
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=mand_facts,
        relevant_chunk_ids=[],  # Empty ground truth
    )
    assert is_verified_4 is False
    assert criteria_4["step_07_ground_truth_chunks_retrieved"] is False, (
        "Absence of ground-truth chunk IDs must fail evidence verification!"
    )


def test_incorrect_boundary_classification_fixed():
    """Verify that a corpus-boundary query returning 'Not in context.' is classified as abstention, NEVER initial_success."""
    state = {
        "question": "What was the exact numerical learning rate value used for the base Transformer?",
        "attempts": 0,
        "verdict": "PASS",
        "draft": "Not in context.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1", "c2"]},
            {"node": "generate_node", "draft": "Not in context."},
            {"node": "judge_node", "verdict": "PASS", "reason": "Data not in corpus."},
        ],
    }

    cls_name, _ = classify_self_correction_case(
        state=state,
        expected_behavior="abstain",
        supporting_facts=[],
        original_question=state["question"],
        expected_decision="abstain",
        mandatory_facts=[],
        category="corpus_boundary",
        answerability="unanswerable",
    )

    assert cls_name == "abstention", f"Corpus boundary must be classified as 'abstention', got '{cls_name}'"
    assert cls_name != "initial_success", "Corpus boundary must NEVER be classified as 'initial_success'!"


def test_zero_denominator_reports_not_applicable():
    """Verify that when cases_requiring_correction == 0, rates report 'N/A'."""
    mini_dataset = [
        {
            "id": "T-01",
            "category": "control",
            "answerability": "answerable",
            "expected_decision": "answer",
            "question": "Q1",
            "expected_answer": "six layers",
            "mandatory_facts": ["six layers"],
            "supporting_facts": ["six layers"],
            "relevant_chunk_ids": ["c1"],
            "correction_reason": "control",
        },
        {
            "id": "T-02",
            "category": "unanswerable",
            "answerability": "unanswerable",
            "expected_decision": "abstain",
            "question": "Q2",
            "expected_answer": "Not in context.",
            "mandatory_facts": [],
            "supporting_facts": [],
            "relevant_chunk_ids": [],
            "correction_reason": "unanswerable",
        },
    ]

    def mock_pipeline(query: str) -> Dict[str, Any]:
        if query == "Q1":
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "It has six layers.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c1"]},
                    {"node": "generate_node", "draft": "six layers"},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }
        else:
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "Not in context.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c2"]},
                    {"node": "generate_node", "draft": "Not in context."},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }

    results = evaluate_self_correction_benchmark(
        dataset=mini_dataset,
        pipeline_func=mock_pipeline,
        sleep_between_queries=0.0,
    )

    summary = results["summary"]
    assert summary["cases_requiring_correction"] == 0
    assert summary["successful_correction_rate_of_required"] == "N/A"
    assert summary["failed_correction_rate"] == "N/A"
    assert summary["max_rewrite_compliance_rate"] == "N/A"
    assert summary["total_live_corrections"] == 0
    assert summary["answer_accuracy"] == 100.0
    assert summary["out_of_scope_abstention_rate"] == 100.0
    assert summary["overall_decision_correctness"] == 100.0
    # 1 abstention out of 2 total evaluated = 50.0%
    assert summary["overall_abstention_rate"] == 50.0


def test_abstention_on_answerable_query_rejected():
    """Verify that returning 'Not in context.' on an answerable query is rejected, not counted as abstention."""
    state_improper_abstention = {
        "question": "How many encoder and decoder layers does the base Transformer use?",
        "attempts": 0,
        "verdict": "PASS",
        "draft": "Not in context.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": "Not in context."},
            {"node": "judge_node", "verdict": "PASS"},
        ],
    }

    cls_name, info = classify_self_correction_case(
        state=state_improper_abstention,
        expected_behavior="initial_success",
        supporting_facts=["six encoder", "six decoder"],
        original_question=state_improper_abstention["question"],
        expected_decision="answer",
        mandatory_facts=["six encoder", "six decoder"],
        answerability="answerable",
    )

    assert cls_name != "abstention", "Abstention on an answerable query must NEVER be classified as 'abstention'"
    assert cls_name == "false_positive_pass", "Judge passing abstention on an answerable query is a false_positive_pass"


def test_missing_mandatory_facts_fails_verification_and_classification():
    """Verify that an answer draft having a single keyword but missing mandatory facts fails."""
    orig_q = "What is the penalty that creates sparse features?"
    mand_facts = ["l1", "regularization", "sparse"]

    # Draft has "l1" and "sparse", but is missing "regularization"
    incomplete_draft = "The L1 penalty produces sparse features."

    state = {
        "question": orig_q,
        "attempts": 0,
        "verdict": "PASS",
        "draft": incomplete_draft,
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": incomplete_draft},
            {"node": "judge_node", "verdict": "PASS"},
        ],
    }

    cls_name, info = classify_self_correction_case(
        state=state,
        expected_decision="answer",
        mandatory_facts=mand_facts,
        original_question=orig_q,
    )

    assert cls_name == "false_positive_pass", "Must be classified as false_positive_pass when mandatory facts are missing"
    assert cls_name != "initial_success"


def test_wrong_subsequent_retrieval_query_fails_verification():
    """Verify that subsequent retrieval using the wrong query fails verification."""
    trace = _build_mock_successful_trace()
    # Change the query used in the second retrieval to the original query
    trace[4]["question"] = "What is the penalty that creates sparse features?"

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question="What is the penalty that creates sparse features?",
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=["l1", "regularization"],
        relevant_chunk_ids=["machine_learning_fundamentals_txt_chunk_0003"],
    )

    assert is_verified is False
    assert criteria["step_06_subsequent_retrieval_uses_rewrite"] is False


def test_invalid_final_judge_ordering_fails_verification():
    """Verify that final Judge PASS occurring before revised Generate fails verification."""
    trace = _build_mock_successful_trace()
    # Move judge before revised generate
    trace[5], trace[6] = trace[6], trace[5]

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question="What is the penalty that creates sparse features?",
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=["l1", "regularization"],
        relevant_chunk_ids=["machine_learning_fundamentals_txt_chunk_0003"],
    )

    assert is_verified is False
    assert criteria["step_09_final_judge_pass_after_generate"] is False


def test_unrelated_chunks_after_rewrite_fails_verification():
    """Verify that fetching different chunks that do NOT contain ground-truth chunks fails."""
    trace = _build_mock_successful_trace()
    trace[4]["chunk_ids"] = ["random_chunk_99", "random_chunk_98"]

    is_verified, criteria = verify_strict_self_correction_trace(
        trace=trace,
        original_question="What is the penalty that creates sparse features?",
        final_verdict="PASS",
        final_draft="L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        mandatory_facts=["l1", "regularization"],
        relevant_chunk_ids=["machine_learning_fundamentals_txt_chunk_0003"],
    )

    assert is_verified is False
    assert criteria["step_07_ground_truth_chunks_retrieved"] is False


def test_mutually_exclusive_classification_logic():
    """Verify that case classification properly partitions states into mutually exclusive classes."""
    mand_facts = ["six encoder", "six decoder"]
    orig_q = "How many layers?"

    # 1. initial_success: 0 attempts, PASS, mandatory facts present
    state_init_succ = {
        "question": orig_q,
        "attempts": 0,
        "verdict": "PASS",
        "draft": "The base model contains six encoder layers and six decoder layers.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": "six encoder and six decoder layers"},
            {"node": "judge_node", "verdict": "PASS"},
        ],
    }
    cls_name, _ = classify_self_correction_case(
        state=state_init_succ,
        expected_decision="answer",
        mandatory_facts=mand_facts,
        original_question=orig_q,
    )
    assert cls_name == "initial_success"

    # 2. successful_correction: valid 10-step trace
    state_corr_succ = {
        "question": "What is the penalty that creates sparse features?",
        "attempts": 1,
        "verdict": "PASS",
        "draft": "L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
        "trace": _build_mock_successful_trace(),
    }
    cls_name, _ = classify_self_correction_case(
        state=state_corr_succ,
        expected_decision="answer",
        mandatory_facts=["l1", "regularization"],
        relevant_chunk_ids=["machine_learning_fundamentals_txt_chunk_0003"],
        original_question="What is the penalty that creates sparse features?",
    )
    assert cls_name == "successful_correction"

    # 3. failed_correction: attempts >= 2, final verdict FAIL
    state_failed = {
        "question": orig_q,
        "attempts": 2,
        "verdict": "FAIL",
        "draft": "Unsupported answer draft.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": "draft 1"},
            {"node": "judge_node", "verdict": "FAIL"},
            {"node": "rewrite_node", "attempt": 1, "rewritten_question": "q1"},
            {"node": "retrieve_node", "chunk_ids": ["c2"]},
            {"node": "generate_node", "draft": "draft 2"},
            {"node": "judge_node", "verdict": "FAIL"},
            {"node": "rewrite_node", "attempt": 2, "rewritten_question": "q2"},
            {"node": "retrieve_node", "chunk_ids": ["c3"]},
            {"node": "generate_node", "draft": "draft 3"},
            {"node": "judge_node", "verdict": "FAIL"},
        ],
    }
    cls_name, _ = classify_self_correction_case(
        state=state_failed,
        expected_decision="answer",
        mandatory_facts=mand_facts,
        original_question=orig_q,
    )
    assert cls_name == "failed_correction"

    # 4. abstention: unanswerable query returning 'Not in context.' with PASS
    state_abstain = {
        "question": "What is the capital of France?",
        "attempts": 0,
        "verdict": "PASS",
        "draft": "Not in context.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": "Not in context."},
            {"node": "judge_node", "verdict": "PASS"},
        ],
    }
    cls_name, _ = classify_self_correction_case(
        state=state_abstain,
        expected_decision="abstain",
        mandatory_facts=[],
        original_question="What is the capital of France?",
    )
    assert cls_name == "abstention"

    # 5. false_positive_pass: unanswerable query where model hallucinated an answer and judge passed
    state_fp_pass = {
        "question": "What is the capital of France?",
        "attempts": 0,
        "verdict": "PASS",
        "draft": "The capital is Paris.",
        "trace": [
            {"node": "retrieve_node", "chunk_ids": ["c1"]},
            {"node": "generate_node", "draft": "The capital is Paris."},
            {"node": "judge_node", "verdict": "PASS"},
        ],
    }
    cls_name, _ = classify_self_correction_case(
        state=state_fp_pass,
        expected_decision="abstain",
        mandatory_facts=[],
        original_question="What is the capital of France?",
    )
    assert cls_name == "false_positive_pass"


def test_extract_attempt_history():
    """Verify that multi-attempt linear traces are cleanly structured into attempt objects."""
    trace = _build_mock_successful_trace()
    history = extract_attempt_history(trace)

    assert "attempt_0" in history
    assert "attempt_1" in history

    # Check attempt 0 details
    assert history["attempt_0"]["retrieved_chunk_ids"] == ["c_gen_01", "c_gen_02", "c_gen_03", "c_gen_04"]
    assert history["attempt_0"]["judge_verdict"] == "FAIL"
    assert history["attempt_0"]["rewritten_query"] is None

    # Check attempt 1 details
    assert history["attempt_1"]["retrieved_chunk_ids"] == [
        "machine_learning_fundamentals_txt_chunk_0003",
        "c_gen_02",
        "c_gen_03",
        "c_gen_04",
    ]
    assert history["attempt_1"]["judge_verdict"] == "PASS"
    assert history["attempt_1"]["rewritten_query"] == "L1 Lasso regularization penalty for sparse feature representations"


def test_benchmark_metrics_calculation_math():
    """Verify quantitative aggregation formulas using deterministic mock outputs."""
    mini_dataset = [
        {
            "id": "T-01",
            "category": "control",
            "answerability": "answerable",
            "expected_decision": "answer",
            "question": "Q1",
            "expected_behavior": "initial_success",
            "expected_answer": "six layers",
            "mandatory_facts": ["six layers"],
            "supporting_facts": ["six layers"],
            "relevant_chunk_ids": ["c1"],
            "correction_reason": "control",
        },
        {
            "id": "T-02",
            "category": "ambiguous",
            "answerability": "answerable",
            "expected_decision": "answer",
            "question": "What is the penalty that creates sparse features?",
            "expected_behavior": "initial_success",
            "expected_answer": "L1 Lasso",
            "mandatory_facts": ["l1", "regularization"],
            "supporting_facts": ["l1", "regularization"],
            "relevant_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0003"],
            "correction_reason": "ambiguous",
        },
        {
            "id": "T-03",
            "category": "corpus_boundary",
            "answerability": "unanswerable",
            "expected_decision": "abstain",
            "question": "Q3",
            "expected_behavior": "abstain",
            "expected_answer": "Not in context.",
            "mandatory_facts": [],
            "supporting_facts": [],
            "relevant_chunk_ids": [],
            "correction_reason": "absent",
        },
        {
            "id": "T-04",
            "category": "unanswerable",
            "answerability": "unanswerable",
            "expected_decision": "abstain",
            "question": "Q4",
            "expected_behavior": "abstain",
            "expected_answer": "Not in context.",
            "mandatory_facts": [],
            "supporting_facts": [],
            "relevant_chunk_ids": [],
            "correction_reason": "unanswerable",
        },
    ]

    def mock_pipeline(query: str) -> Dict[str, Any]:
        if query == "Q1":
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "It has six layers.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c1"]},
                    {"node": "generate_node", "draft": "six layers"},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }
        elif "penalty" in query:
            return {
                "question": query,
                "attempts": 1,
                "verdict": "PASS",
                "draft": "L1 regularization, also known as Lasso regression, introduces a penalty proportional to the absolute values of the weights, producing sparse feature representations.",
                "trace": _build_mock_successful_trace(),
            }
        elif query == "Q3":
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "Not in context.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c1"]},
                    {"node": "generate_node", "draft": "Not in context."},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }
        else:  # Q4
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "Not in context.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c1"]},
                    {"node": "generate_node", "draft": "Not in context."},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }

    results = evaluate_self_correction_benchmark(
        dataset=mini_dataset,
        pipeline_func=mock_pipeline,
        sleep_between_queries=0.0,
    )

    summary = results["summary"]
    assert summary["total_evaluated"] == 4
    assert summary["total_answerable"] == 2
    assert summary["total_unanswerable"] == 2
    assert summary["total_out_of_scope"] == 1
    assert summary["total_corpus_boundary"] == 1
    assert summary["cases_requiring_correction"] == 1  # Q2 (1 retry)

    counts = summary["classification_counts"]
    assert counts["initial_success"] == 1
    assert counts["successful_correction"] == 1
    assert counts["failed_correction"] == 0
    assert counts["abstention"] == 2

    # Answer accuracy = 2 / 2 = 100.0%
    assert summary["answer_accuracy"] == 100.0
    # First attempt success rate = 1 / 2 = 50.0%
    assert summary["first_attempt_success_rate"] == 50.0
    # Successful correction of required = 1 / 1 = 100.0%
    assert summary["successful_correction_rate_of_required"] == 100.0
    # Failed correction rate = 0 / 1 = 0.0%
    assert summary["failed_correction_rate"] == 0.0
    # Out of scope abstention rate = 1 / 1 = 100.0%
    assert summary["out_of_scope_abstention_rate"] == 100.0
    # Corpus boundary abstention rate = 1 / 1 = 100.0%
    assert summary["corpus_boundary_abstention_rate"] == 100.0
    # Overall abstention rate = 2 / 4 = 50.0%
    assert summary["overall_abstention_rate"] == 50.0
    # Overall decision correctness = 4 / 4 = 100.0%
    assert summary["overall_decision_correctness"] == 100.0
    # Max rewrite compliance rate = 4 / 4 = 100.0% (since retries occurred)
    assert summary["max_rewrite_compliance_rate"] == 100.0


def test_final_audit_metric_specifications():
    """Verify exact Step 3 Final Metric Audit requirements against full 16-case mock execution."""
    # Mock pipeline returning attempt 0 results matching the 16 live benchmark cases
    def mock_full_pipeline(query: str) -> Dict[str, Any]:
        matching_case = next((c for c in SELF_CORRECTION_DATASET if c["question"] == query), None)
        if matching_case and matching_case["expected_decision"] == "abstain":
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": "Not in context.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": ["c1"]},
                    {"node": "generate_node", "draft": "Not in context."},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }
        elif matching_case:
            facts_str = " ".join(matching_case.get("mandatory_facts", []))
            return {
                "question": query,
                "attempts": 0,
                "verdict": "PASS",
                "draft": f"The answer contains {facts_str}.",
                "trace": [
                    {"node": "retrieve_node", "chunk_ids": matching_case.get("relevant_chunk_ids", ["c1"])},
                    {"node": "generate_node", "draft": f"The answer contains {facts_str}."},
                    {"node": "judge_node", "verdict": "PASS"},
                ],
            }
        return {
            "question": query,
            "attempts": 0,
            "verdict": "PASS",
            "draft": "Grounded answer.",
            "trace": [],
        }

    results = evaluate_self_correction_benchmark(
        dataset=SELF_CORRECTION_DATASET,
        pipeline_func=mock_full_pipeline,
        sleep_between_queries=0.0,
    )
    summary = results["summary"]

    # 1. Overall abstention rate must be 25.0% (4/16)
    assert summary["overall_abstention_rate"] == 25.0
    # 2. Out-of-scope abstention rate remains 100% (2/2)
    assert summary["out_of_scope_abstention_rate"] == 100.0
    # 3. Corpus-boundary abstention rate remains 100% (2/2)
    assert summary["corpus_boundary_abstention_rate"] == 100.0
    # 4. Max rewrite compliance rate must be N/A because zero rewrites occurred
    assert summary["max_rewrite_compliance_rate"] == "N/A"
    # 5. Failed correction rate must be N/A when no cases required correction
    assert summary["failed_correction_rate"] == "N/A"
    # 6. Keep successful correction rate of required as N/A
    assert summary["successful_correction_rate_of_required"] == "N/A"
    # 7. Preserve total live corrections = 0
    assert summary["total_live_corrections"] == 0
    # 8. Preserve average rewrites per query = 0.0
    assert summary["average_rewrites_per_query"] == 0.0
    # 9. Answer accuracy preserved at 100.0% (12/12)
    assert summary["answer_accuracy"] == 100.0
    # 10. Overall decision correctness preserved at 100.0% (16/16)
    assert summary["overall_decision_correctness"] == 100.0
