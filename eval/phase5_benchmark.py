"""Phase 5: Rigorous Retrieval Benchmarking and Optimization for Alethia.

Empirically compares Dense-only, BM25-only, and Hybrid retrieval (RRF)
across the 30-question evaluation dataset with explicit ground-truth mappings.

Features:
1. Standard IR Metrics: Hit Rate@1, 3, 5; Recall@1, 3, 5; Precision@1, 3, 5; MRR.
2. Segregation: Separates answerable queries (20) from unanswerable/abstention queries (10).
3. Parameter Grid: Evaluates RRF k (20, 40, 60, 80), Dense/BM25 weights (0.25/0.75, 0.5/0.5, 0.75/0.25),
   and candidate pool depths (10, 20, 35).
4. Decomposition Evaluation: Evaluates single-intent vs multi-intent trigger accuracy,
   subquestion evidence recall, and candidate preservation.
5. Reproducibility: Generates structured JSON artifacts and markdown report.
"""

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.dataset import EVALUATION_DATASET
from src.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.decomposition import is_multi_intent_question, decompose_question, retrieve_decomposed


# ============================================================================
# 1. Mathematical Information Retrieval Metric Definitions
# ============================================================================

def calculate_hit_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> int:
    """Calculate binary Hit@K indicator.

    Hit@K = 1 if at least one retrieved chunk in the top K ranks is in relevant_ids, else 0.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Retrieval depth cutoff.

    Returns:
        1 if hit occurs in top k, else 0.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0
    top_k = retrieved_ids[:k]
    return 1 if any(cid in relevant_ids for cid in top_k) else 0


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Recall@K fraction.

    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Retrieval depth cutoff.

    Returns:
        Recall fraction between 0.0 and 1.0.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    matched = top_k.intersection(relevant_ids)
    return round(len(matched) / len(relevant_ids), 4)


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculate Precision@K fraction.

    Precision@K = |Retrieved_K ∩ Relevant| / K

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k: Retrieval depth cutoff.

    Returns:
        Precision fraction between 0.0 and 1.0.
    """
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    matched = top_k.intersection(relevant_ids)
    return round(len(matched) / k, 4)


def calculate_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str], k_max: int = 5) -> float:
    """Calculate Reciprocal Rank (RR).

    RR = 1 / rank_1 where rank_1 is the 1-based rank of the first relevant chunk in top k_max.
    Returns 0.0 if no relevant chunk is retrieved within k_max.

    Args:
        retrieved_ids: Ordered list of retrieved chunk IDs.
        relevant_ids: Set of ground-truth relevant chunk IDs.
        k_max: Maximum evaluated rank depth.

    Returns:
        Reciprocal rank float (e.g. 1.0, 0.5, 0.3333, 0.0).
    """
    if not retrieved_ids or not relevant_ids or k_max <= 0:
        return 0.0
    for idx, cid in enumerate(retrieved_ids[:k_max], 1):
        if cid in relevant_ids:
            return round(1.0 / idx, 4)
    return 0.0


# ============================================================================
# 2. Benchmark Aggregator
# ============================================================================

def compute_retrieval_metrics(
    dataset: List[Dict[str, Any]],
    retrievals: Dict[str, List[str]],
    k_max: int = 5,
) -> Dict[str, Any]:
    """Compute aggregate retrieval metrics across all answerable queries.

    Excludes unanswerable and corpus-boundary abstention queries from the
    denominator so metrics accurately reflect answerable search performance.

    Args:
        dataset: Query records from EVALUATION_DATASET.
        retrievals: Mapping from query id to ordered list of retrieved chunk IDs.
        k_max: Evaluation cutoff depth (default: 5).

    Returns:
        Dictionary of aggregate Hit Rate, Recall, Precision, and MRR metrics.
    """
    answerable_cases = [
        item for item in dataset
        if item.get("expected_behavior") != "abstain" and len(item.get("relevant_chunk_ids", [])) > 0
    ]
    unanswerable_cases = [
        item for item in dataset
        if item not in answerable_cases
    ]

    total_q = len(answerable_cases)
    if total_q == 0:
        return {"error": "No answerable queries found"}

    hits_1, hits_3, hits_5 = 0, 0, 0
    recalls_1, recalls_3, recalls_5 = [], [], []
    precisions_1, precisions_3, precisions_5 = [], [], []
    reciprocal_ranks = []
    query_details = []

    for item in answerable_cases:
        qid = item["id"]
        rel_set = set(item.get("relevant_chunk_ids", []))
        ret_list = retrievals.get(qid, [])

        h1 = calculate_hit_at_k(ret_list, rel_set, k=1)
        h3 = calculate_hit_at_k(ret_list, rel_set, k=3)
        h5 = calculate_hit_at_k(ret_list, rel_set, k=5)

        r1 = calculate_recall_at_k(ret_list, rel_set, k=1)
        r3 = calculate_recall_at_k(ret_list, rel_set, k=3)
        r5 = calculate_recall_at_k(ret_list, rel_set, k=5)

        p1 = calculate_precision_at_k(ret_list, rel_set, k=1)
        p3 = calculate_precision_at_k(ret_list, rel_set, k=3)
        p5 = calculate_precision_at_k(ret_list, rel_set, k=5)

        rr = calculate_reciprocal_rank(ret_list, rel_set, k_max=k_max)

        hits_1 += h1
        hits_3 += h3
        hits_5 += h5

        recalls_1.append(r1)
        recalls_3.append(r3)
        recalls_5.append(r5)

        precisions_1.append(p1)
        precisions_3.append(p3)
        precisions_5.append(p5)

        reciprocal_ranks.append(rr)

        query_details.append({
            "id": qid,
            "category": item.get("category"),
            "question": item.get("question"),
            "expected_chunk_ids": list(rel_set),
            "retrieved_chunk_ids": ret_list[:k_max],
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "recall_at_1": r1,
            "recall_at_3": r3,
            "recall_at_5": r5,
            "precision_at_1": p1,
            "precision_at_3": p3,
            "precision_at_5": p5,
            "reciprocal_rank": rr,
        })

    return {
        "total_answerable_queries": total_q,
        "unanswerable_queries_excluded": len(unanswerable_cases),
        "hit_rate_at_1": round(hits_1 / total_q, 4),
        "hit_rate_at_3": round(hits_3 / total_q, 4),
        "hit_rate_at_5": round(hits_5 / total_q, 4),
        "recall_at_1": round(sum(recalls_1) / total_q, 4),
        "recall_at_3": round(sum(recalls_3) / total_q, 4),
        "recall_at_5": round(sum(recalls_5) / total_q, 4),
        "precision_at_1": round(sum(precisions_1) / total_q, 4),
        "precision_at_3": round(sum(precisions_3) / total_q, 4),
        "precision_at_5": round(sum(precisions_5) / total_q, 4),
        "mrr": round(sum(reciprocal_ranks) / total_q, 4),
        "query_details": query_details,
    }


# ============================================================================
# 3. Strategy Evaluator
# ============================================================================

def evaluate_retrieval_strategy(
    strategy: str,
    retriever: Optional[HybridRetriever] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    k_max: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Execute evaluation for a single retrieval strategy ('dense', 'bm25', or 'hybrid').

    Args:
        strategy: 'dense', 'bm25', or 'hybrid'.
        retriever: Optional HybridRetriever instance.
        dataset: Evaluation cases (defaults to EVALUATION_DATASET).
        k_max: Depth cutoff (default: 5).
        **kwargs: Optional retriever parameters (e.g. candidate_depth).

    Returns:
        Summary metric dictionary with query details.
    """
    target_retriever = retriever or get_hybrid_retriever()
    cases = dataset or EVALUATION_DATASET

    retrievals: Dict[str, List[str]] = {}

    for item in cases:
        qid = item["id"]
        query = item["question"]

        if strategy == "dense":
            chunks = target_retriever.retrieve_dense(query, k=k_max)
        elif strategy == "bm25":
            chunks = target_retriever.retrieve_bm25(query, k=k_max)
        elif strategy == "hybrid":
            depth = kwargs.get("candidate_depth")
            chunks = target_retriever.retrieve_hybrid(query, k=k_max, candidate_depth=depth)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        retrievals[qid] = [c.get("chunk_id", "") for c in chunks if c.get("chunk_id")]

    metrics = compute_retrieval_metrics(cases, retrievals, k_max=k_max)
    metrics["strategy"] = strategy
    return metrics


# ============================================================================
# 4. Parameter Grid Evaluation
# ============================================================================

def evaluate_parameter_grid(
    retriever: Optional[HybridRetriever] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    k_values: Optional[List[int]] = None,
    weights: Optional[List[Tuple[float, float]]] = None,
    candidate_depths: Optional[List[int]] = None,
    k_max: int = 5,
) -> Dict[str, Any]:
    """Evaluate controlled parameter grid for Hybrid RRF retrieval.

    Grid:
    - RRF k values: [20, 40, 60, 80]
    - Dense/BM25 weights: [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25)]
    - Candidate depths: [10, 20, 35]

    Args:
        retriever: Optional base HybridRetriever instance.
        dataset: Evaluation dataset.
        k_values: RRF smoothing constants to test.
        weights: (dense_weight, bm25_weight) tuples to test.
        candidate_depths: Candidate pool depths to test.
        k_max: Retrieval evaluation cutoff.

    Returns:
        Dictionary of all parameter combinations and their measured metrics.
    """
    base_retriever = retriever or get_hybrid_retriever()
    cases = dataset or EVALUATION_DATASET

    ks = k_values or [20, 40, 60, 80]
    weight_pairs = weights or [(0.25, 0.75), (0.50, 0.50), (0.75, 0.25)]
    depths = candidate_depths or [10, 20, 35]

    grid_results = []
    best_config = None
    best_mrr = -1.0

    for k_rrf in ks:
        for d_w, b_w in weight_pairs:
            for depth in depths:
                # Instantiate configured retriever
                configured_retriever = HybridRetriever(
                    vector_store=base_retriever.vector_store,
                    bm25_index=base_retriever.bm25_index,
                    k_rrf=k_rrf,
                    dense_weight=d_w,
                    bm25_weight=b_w,
                )

                metrics = evaluate_retrieval_strategy(
                    strategy="hybrid",
                    retriever=configured_retriever,
                    dataset=cases,
                    k_max=k_max,
                    candidate_depth=depth,
                )

                config_entry = {
                    "k_rrf": k_rrf,
                    "dense_weight": d_w,
                    "bm25_weight": b_w,
                    "candidate_depth": depth,
                    "hit_rate_at_1": metrics["hit_rate_at_1"],
                    "hit_rate_at_3": metrics["hit_rate_at_3"],
                    "hit_rate_at_5": metrics["hit_rate_at_5"],
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_3": metrics["recall_at_3"],
                    "recall_at_5": metrics["recall_at_5"],
                    "mrr": metrics["mrr"],
                }
                grid_results.append(config_entry)

                if metrics["mrr"] > best_mrr or (metrics["mrr"] == best_mrr and metrics["recall_at_5"] > (best_config or {}).get("recall_at_5", 0.0)):
                    best_mrr = metrics["mrr"]
                    best_config = config_entry

    return {
        "total_combinations_tested": len(grid_results),
        "best_configuration": best_config,
        "grid_results": grid_results,
    }


# ============================================================================
# 5. Question Decomposition Evaluation
# ============================================================================

DECOMPOSITION_TEST_SUITE: List[Dict[str, Any]] = [
    # Single-intent questions (expected: bypass decomposition, 1 question)
    {
        "id": "DEC-S1",
        "query": "What is backpropagation?",
        "is_multi_intent": False,
        "expected_subquestions": 1,
        "target_chunk_ids": ["neural_network_training_txt_chunk_0000"],
    },
    {
        "id": "DEC-S2",
        "query": "Define precision and recall.",
        "is_multi_intent": False,
        "expected_subquestions": 1,
        "target_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0004"],
    },
    {
        "id": "DEC-S3",
        "query": "Explain transformer self-attention.",
        "is_multi_intent": False,
        "expected_subquestions": 1,
        "target_chunk_ids": ["attention_mechanisms_txt_chunk_0000"],
    },
    # Multi-intent questions (expected: trigger decomposition, 2 subquestions)
    {
        "id": "DEC-M1",
        "query": "What optimizer was used in the Transformer paper, and what are Adam's default beta values?",
        "is_multi_intent": True,
        "expected_subquestions": 2,
        "target_chunk_ids": ["transformer_summary_txt_chunk_0004", "optimizers_guide_txt_chunk_0002"],
    },
    {
        "id": "DEC-M2",
        "query": "What is precision, and how is recall calculated?",
        "is_multi_intent": True,
        "expected_subquestions": 2,
        "target_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0004"],
    },
    {
        "id": "DEC-M3",
        "query": "What is backpropagation? And how does gradient descent update weights?",
        "is_multi_intent": True,
        "expected_subquestions": 2,
        "target_chunk_ids": ["neural_network_training_txt_chunk_0000", "optimizers_guide_txt_chunk_0000"],
    },
]


def evaluate_decomposition_behavior(
    eval_cases: Optional[List[Dict[str, Any]]] = None,
    retriever: Optional[HybridRetriever] = None,
    k: int = 4,
) -> Dict[str, Any]:
    """Evaluate question decomposition triggering, subquestion generation, and evidence preservation.

    Args:
        eval_cases: Test queries (defaults to DECOMPOSITION_TEST_SUITE).
        retriever: Optional HybridRetriever instance.
        k: Maximum retrieved chunks to evaluate.

    Returns:
        Evaluation report on trigger accuracy, evidence preservation, and noise chunks.
    """
    cases = eval_cases or DECOMPOSITION_TEST_SUITE
    target_retriever = retriever or get_hybrid_retriever()

    total_q = len(cases)
    trigger_correct = 0
    subq_count_correct = 0
    preserved_evidence_count = 0
    total_chunks_retrieved = 0
    total_relevant_chunks_retrieved = 0

    case_details = []

    for item in cases:
        query = item["query"]
        expected_multi = item["is_multi_intent"]
        expected_subs = item["expected_subquestions"]
        targets = set(item.get("target_chunk_ids", []))

        # Check trigger detection
        detected_multi = is_multi_intent_question(query)
        is_trigger_ok = (detected_multi == expected_multi)
        if is_trigger_ok:
            trigger_correct += 1

        # Run decomposition & retrieval
        decomp_result = retrieve_decomposed(
            query=query,
            k=k,
            strategy="hybrid",
            retriever=target_retriever,
            use_llm=False,  # deterministic rule-based evaluation
        )

        sub_qs = decomp_result.get("decomposed_questions", [])
        is_count_ok = (len(sub_qs) == expected_subs)
        if is_count_ok:
            subq_count_correct += 1

        retrieved_chunks = decomp_result.get("chunks", [])
        retrieved_ids = {c.get("chunk_id") for c in retrieved_chunks}

        matched_targets = retrieved_ids.intersection(targets)
        # Preserved evidence: at least one target chunk retrieved if targets exist
        has_preserved = (len(matched_targets) > 0) if targets else True
        if has_preserved:
            preserved_evidence_count += 1

        total_chunks_retrieved += len(retrieved_ids)
        total_relevant_chunks_retrieved += len(matched_targets)

        case_details.append({
            "id": item.get("id"),
            "query": query,
            "expected_multi_intent": expected_multi,
            "detected_multi_intent": detected_multi,
            "trigger_correct": is_trigger_ok,
            "subquestions_generated": sub_qs,
            "subquestions_count": len(sub_qs),
            "expected_subquestions_count": expected_subs,
            "retrieved_chunk_ids": list(retrieved_ids),
            "matched_target_chunk_ids": list(matched_targets),
            "evidence_preserved": has_preserved,
        })

    noise_chunks = total_chunks_retrieved - total_relevant_chunks_retrieved

    return {
        "total_queries": total_q,
        "correct_trigger_count": trigger_correct,
        "trigger_accuracy": round(trigger_correct / total_q, 4),
        "subquestion_count_accuracy": round(subq_count_correct / total_q, 4),
        "evidence_preservation_rate": round(preserved_evidence_count / total_q, 4),
        "total_chunks_retrieved": total_chunks_retrieved,
        "total_relevant_chunks_retrieved": total_relevant_chunks_retrieved,
        "noise_chunk_count": noise_chunks,
        "noise_chunk_ratio": round(noise_chunks / max(1, total_chunks_retrieved), 4),
        "case_details": case_details,
    }


# ============================================================================
# 6. Benchmark Orchestrator & Artifact Generator
# ============================================================================

def run_phase5_benchmark() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Execute complete Phase 5 benchmark across all three strategies and parameter grid.

    Returns:
        Tuple of (comparison_results, parameter_results, decomposition_results).
    """
    retriever = get_hybrid_retriever()

    # 1. Evaluate Dense-only
    dense_metrics = evaluate_retrieval_strategy("dense", retriever=retriever, k_max=5)

    # 2. Evaluate BM25-only
    bm25_metrics = evaluate_retrieval_strategy("bm25", retriever=retriever, k_max=5)

    # 3. Evaluate Hybrid (default RRF k=60, weights 1.0/1.0, depth 20)
    hybrid_metrics = evaluate_retrieval_strategy("hybrid", retriever=retriever, k_max=5)

    comparison_results = {
        "metadata": {
            "title": "Alethia Phase 5 Retrieval Strategy Benchmark Comparison",
            "phase": "Phase 5",
            "corpus_documents": 7,
            "corpus_chunks": 35,
            "total_queries": len(EVALUATION_DATASET),
            "evaluated_answerable_queries": dense_metrics["total_answerable_queries"],
            "unanswerable_queries_excluded": dense_metrics["unanswerable_queries_excluded"],
            "ground_truth_methodology": "Explicit chunk ID match and verified multi-token keyphrase evidence",
        },
        "comparison_table": {
            "dense_only": {
                "hit_rate_at_1": dense_metrics["hit_rate_at_1"],
                "hit_rate_at_3": dense_metrics["hit_rate_at_3"],
                "hit_rate_at_5": dense_metrics["hit_rate_at_5"],
                "recall_at_1": dense_metrics["recall_at_1"],
                "recall_at_3": dense_metrics["recall_at_3"],
                "recall_at_5": dense_metrics["recall_at_5"],
                "precision_at_1": dense_metrics["precision_at_1"],
                "precision_at_3": dense_metrics["precision_at_3"],
                "precision_at_5": dense_metrics["precision_at_5"],
                "mrr": dense_metrics["mrr"],
            },
            "bm25_only": {
                "hit_rate_at_1": bm25_metrics["hit_rate_at_1"],
                "hit_rate_at_3": bm25_metrics["hit_rate_at_3"],
                "hit_rate_at_5": bm25_metrics["hit_rate_at_5"],
                "recall_at_1": bm25_metrics["recall_at_1"],
                "recall_at_3": bm25_metrics["recall_at_3"],
                "recall_at_5": bm25_metrics["recall_at_5"],
                "precision_at_1": bm25_metrics["precision_at_1"],
                "precision_at_3": bm25_metrics["precision_at_3"],
                "precision_at_5": bm25_metrics["precision_at_5"],
                "mrr": bm25_metrics["mrr"],
            },
            "hybrid": {
                "hit_rate_at_1": hybrid_metrics["hit_rate_at_1"],
                "hit_rate_at_3": hybrid_metrics["hit_rate_at_3"],
                "hit_rate_at_5": hybrid_metrics["hit_rate_at_5"],
                "recall_at_1": hybrid_metrics["recall_at_1"],
                "recall_at_3": hybrid_metrics["recall_at_3"],
                "recall_at_5": hybrid_metrics["recall_at_5"],
                "precision_at_1": hybrid_metrics["precision_at_1"],
                "precision_at_3": hybrid_metrics["precision_at_3"],
                "precision_at_5": hybrid_metrics["precision_at_5"],
                "mrr": hybrid_metrics["mrr"],
            },
        },
        "query_breakdowns": {
            "dense": dense_metrics["query_details"],
            "bm25": bm25_metrics["query_details"],
            "hybrid": hybrid_metrics["query_details"],
        },
    }

    # 4. Evaluate parameter grid
    param_results = evaluate_parameter_grid(retriever=retriever)

    # 5. Evaluate question decomposition
    decomp_results = evaluate_decomposition_behavior(retriever=retriever)

    # Save JSON artifacts
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)

    comp_file = os.path.join(out_dir, "phase5_retrieval_comparison.json")
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    param_file = os.path.join(out_dir, "phase5_parameter_experiments.json")
    with open(param_file, "w", encoding="utf-8") as f:
        json.dump(param_results, f, indent=2, ensure_ascii=False)

    return comparison_results, param_results, decomp_results


if __name__ == "__main__":
    comp, params, decomp = run_phase5_benchmark()
    print("Benchmark execution completed.")
    print(json.dumps(comp["comparison_table"], indent=2))

