"""Retrieval quality evaluation module for Alethia (Step 2 Realignment & Audit).

Computes defensible information retrieval metrics across the evaluation dataset:
- Hit Rate@K (Hit@K): Fraction of queries with at least one relevant chunk in top K.
- Recall@K: Fraction of total relevant chunks retrieved in top K, averaged across queries.
- Mean Reciprocal Rank (MRR): Based on the rank of the first relevant result.
- Average Cosine Similarity of top-1 chunk.

Relevance Ground Truth:
Replaces arbitrary substring matching with explicit deterministic chunk IDs and
rigorous multi-token keyphrase evidence criteria. Prevents false-positive
matches caused by single characters, digits, or common generic words.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.store import VectorStore, get_vector_store
from eval.dataset import EVALUATION_DATASET

# Stopwords and overly generic technical terms that must never establish relevance on their own
GENERIC_TERMS = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "q", "k", "v", "a", "an", "the", "in", "on", "at", "for", "of", "with",
    "model", "data", "learning", "network", "system", "layer", "layers",
    "step", "steps", "value", "values", "score", "scores", "rate", "decay",
}


def is_chunk_relevant(
    chunk: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[bool, str]:
    """Determine whether a retrieved chunk is genuinely relevant to an evaluation query.

    A retrieved chunk is relevant if and only if:
    1. Explicit Chunk ID Match:
       The chunk's chunk_id is explicitly listed in ground_truth['relevant_chunk_ids'].
    2. Multi-Token Keyphrase Evidence Match:
       The chunk's source matches ground_truth['relevant_source'] AND its text content
       contains at least one approved multi-token required keyphrase.

    Guards enforced:
    - Never uses single-character or single-digit tokens (e.g. '1', '6', 'q', 'k').
    - Never matches generic words or common stopwords in isolation (e.g. 'decay', 'model').
    - Never marks a chunk relevant merely because it originates from the same source document.
    - Rejects unanswerable / out-of-scope queries (relevant_source is None).

    Args:
        chunk: Retrieved chunk dictionary with keys 'chunk_id', 'source', 'text'.
        ground_truth: Evaluation dataset record for the query.

    Returns:
        Tuple of (is_relevant: bool, match_reason: str).
    """
    expected_src = ground_truth.get("relevant_source")
    if not expected_src or ground_truth.get("expected_behavior") == "abstain":
        return False, "unanswerable_query"

    chunk_id = chunk.get("chunk_id", "")
    chunk_src = chunk.get("source", "")
    chunk_text = chunk.get("text", "")

    # 1. Primary Check: Explicit Relevant Chunk ID Match
    expected_ids = ground_truth.get("relevant_chunk_ids") or []
    if chunk_id and chunk_id in expected_ids:
        return True, f"chunk_id:{chunk_id}"

    # 2. Secondary Check: Multi-Token Keyphrase Evidence
    # Requires matching the expected document source AND a validated multi-token keyphrase
    if chunk_src == expected_src:
        text_lower = chunk_text.lower()
        keyphrases = ground_truth.get("required_keyphrases") or []

        for kp in keyphrases:
            kp_clean = kp.strip().lower()
            if not kp_clean or kp_clean in GENERIC_TERMS:
                continue

            words = kp_clean.split()
            # Enforce multi-token evidence: at least 2 words or long specific identifier (>= 7 chars)
            # and reject single digits or single generic tokens
            if (len(words) >= 2 or len(kp_clean) >= 7) and not kp_clean.isdigit():
                if kp_clean in text_lower:
                    return True, f"keyphrase:{kp_clean[:30]}"

    return False, "no_match"


def evaluate_retrieval(
    store: Optional[VectorStore] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    k_max: int = 5,
) -> Dict[str, Any]:
    """Evaluate retrieval quality across all in-scope answerable questions in the dataset.

    Computes both:
    - Hit Rate@K: queries with at least one relevant chunk in top K / total answerable queries
    - Recall@K: relevant chunks retrieved in top K / total relevant chunks for that query,
                averaged across all answerable queries.
    - MRR: Mean Reciprocal Rank based on the first relevant result.

    Args:
        store: Optional VectorStore instance (defaults to persistent singleton).
        dataset: Evaluation cases (defaults to EVALUATION_DATASET).
        k_max: Maximum retrieval depth for evaluation (default: 5).

    Returns:
        Dict containing summary metrics and per-query breakdown.
    """
    target_store = store or get_vector_store()
    cases = dataset or EVALUATION_DATASET

    # Segregate answerable queries from unanswerable / out-of-scope queries
    eval_cases = [
        c for c in cases
        if c.get("relevant_source") is not None and c.get("expected_behavior") != "abstain"
    ]
    out_of_scope_count = len(cases) - len(eval_cases)
    total_queries = len(eval_cases)

    if total_queries == 0:
        return {
            "error": "No in-scope answerable queries with ground truth found.",
            "total_queries_evaluated": 0,
            "unanswerable_queries_excluded": out_of_scope_count,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_3": 0.0,
            "hit_rate_at_5": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "query_results": [],
        }

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    per_query_recall_1: List[float] = []
    per_query_recall_3: List[float] = []
    per_query_recall_5: List[float] = []

    reciprocal_ranks: List[float] = []
    query_results: List[Dict[str, Any]] = []

    for item in eval_cases:
        qid = item["id"]
        category = item["category"]
        query = item["question"]
        expected_src = item["relevant_source"]
        expected_ids = item.get("relevant_chunk_ids", [])
        total_relevant = max(1, len(expected_ids))

        # Retrieve top k_max chunks from ChromaDB
        retrieved = target_store.retrieve_chunks(query, k=k_max)

        rank_found: Optional[int] = None
        first_matched_chunk_id: Optional[str] = None
        first_match_reason: str = "no_match"
        all_matched_ids: List[str] = []
        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        # Evaluate relevance of each retrieved chunk
        chunk_judgments: List[Tuple[bool, str, Dict[str, Any]]] = []
        for idx, chunk in enumerate(retrieved, 1):
            is_relevant, reason = is_chunk_relevant(chunk, item)
            chunk_judgments.append((is_relevant, reason, chunk))
            if is_relevant:
                all_matched_ids.append(chunk.get("chunk_id", ""))
                if rank_found is None:
                    rank_found = idx
                    first_matched_chunk_id = chunk.get("chunk_id")
                    first_match_reason = reason

        # Count relevant chunks retrieved at each depth K
        rel_at_1 = sum(1 for is_rel, _, _ in chunk_judgments[:1] if is_rel)
        rel_at_3 = sum(1 for is_rel, _, _ in chunk_judgments[:3] if is_rel)
        rel_at_5 = sum(1 for is_rel, _, _ in chunk_judgments[:5] if is_rel)

        # 1. Hit Rate@K: binary indicator (at least one relevant chunk in top K)
        h1 = 1 if rel_at_1 >= 1 else 0
        h3 = 1 if rel_at_3 >= 1 else 0
        h5 = 1 if rel_at_5 >= 1 else 0

        hits_at_1 += h1
        hits_at_3 += h3
        hits_at_5 += h5

        # 2. Recall@K: fraction of total relevant chunks retrieved in top K
        rec1 = rel_at_1 / total_relevant
        rec3 = rel_at_3 / total_relevant
        rec5 = rel_at_5 / total_relevant

        per_query_recall_1.append(rec1)
        per_query_recall_3.append(rec3)
        per_query_recall_5.append(rec5)

        # 3. MRR based on rank of first relevant result
        rr = (1.0 / rank_found) if rank_found is not None else 0.0
        reciprocal_ranks.append(rr)

        query_results.append({
            "id": qid,
            "category": category,
            "query": query,
            "expected_source": expected_src,
            "expected_chunk_ids": expected_ids,
            "total_relevant_chunks": total_relevant,
            "relevant_retrieved_at_1": rel_at_1,
            "relevant_retrieved_at_3": rel_at_3,
            "relevant_retrieved_at_5": rel_at_5,
            "recall_at_1": round(rec1, 4),
            "recall_at_3": round(rec3, 4),
            "recall_at_5": round(rec5, 4),
            "hit_at_1": h1,
            "hit_at_3": h3,
            "hit_at_5": h5,
            "rank_found": rank_found,
            "matched_chunk_id": first_matched_chunk_id,
            "match_reason": first_match_reason,
            "all_matched_chunk_ids": all_matched_ids,
            "top_similarity": top_similarity,
            "rr": round(rr, 4),
        })

    # Summary metrics across all answerable queries
    hit_rate_at_1 = round(hits_at_1 / total_queries, 4)
    hit_rate_at_3 = round(hits_at_3 / total_queries, 4)
    hit_rate_at_5 = round(hits_at_5 / total_queries, 4)

    recall_at_1 = round(sum(per_query_recall_1) / total_queries, 4)
    recall_at_3 = round(sum(per_query_recall_3) / total_queries, 4)
    recall_at_5 = round(sum(per_query_recall_5) / total_queries, 4)

    mrr = round(sum(reciprocal_ranks) / total_queries, 4)

    return {
        "total_queries_evaluated": total_queries,
        "unanswerable_queries_excluded": out_of_scope_count,
        "evaluation_methodology": "explicit_chunk_id_and_multitoken_evidence",
        # Hit Rate@K: queries with at least one relevant chunk in top K / total answerable queries
        "hit_rate_at_1": hit_rate_at_1,
        "hit_rate_at_3": hit_rate_at_3,
        "hit_rate_at_5": hit_rate_at_5,
        # Recall@K: relevant chunks retrieved in top K / total relevant chunks for query, averaged over queries
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        # Mean Reciprocal Rank (MRR): 1 / rank of first relevant chunk
        "mrr": mrr,
        "query_results": query_results,
    }


def format_retrieval_report(results: Dict[str, Any]) -> str:
    """Format retrieval evaluation results into a clean markdown table."""
    total_q = results['total_queries_evaluated']
    lines = [
        "### Information Retrieval Benchmark Summary (Step 2 Realignment & Final Audit)",
        f"- **Evaluated Answerable Queries**: {total_q} (Excludes {results.get('unanswerable_queries_excluded', 0)} out-of-scope abstention queries)",
        f"- **Ground Truth Methodology**: Explicit Chunk IDs & Multi-Token Keyphrase Evidence",
        "",
        "#### Summary Metrics:",
        f"- **Hit Rate@1**: {results['hit_rate_at_1'] * 100:.1f}% ({int(round(results['hit_rate_at_1'] * total_q))}/{total_q}) (Queries with $\\ge 1$ relevant chunk at Rank 1)",
        f"- **Hit Rate@3**: {results['hit_rate_at_3'] * 100:.1f}% ({int(round(results['hit_rate_at_3'] * total_q))}/{total_q}) (Queries with $\\ge 1$ relevant chunk in Top 3)",
        f"- **Hit Rate@5**: {results['hit_rate_at_5'] * 100:.1f}% ({int(round(results['hit_rate_at_5'] * total_q))}/{total_q}) (Queries with $\\ge 1$ relevant chunk in Top 5)",
        f"- **Recall@1**:   {results['recall_at_1'] * 100:.1f}% (Mean fraction of relevant chunks retrieved at Rank 1)",
        f"- **Recall@3**:   {results['recall_at_3'] * 100:.1f}% (Mean fraction of relevant chunks retrieved in Top 3)",
        f"- **Recall@5**:   {results['recall_at_5'] * 100:.1f}% (Mean fraction of relevant chunks retrieved in Top 5)",
        f"- **Mean Reciprocal Rank (MRR)**: {results['mrr']:.4f} (Based on rank of first relevant result)",
        "",
        "| ID | Category | Query Excerpt | Total Rel | Found Rank | Top Sim | Hit@1 | Rec@1 | MRR | First Match Chunk ID |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for q in results.get("query_results", []):
        rank_str = f"Rank {q['rank_found']}" if q['rank_found'] else "Miss (>5)"
        lines.append(
            f"| `{q['id']}` | {q['category']} | {q['query'][:32]}... | {q['total_relevant_chunks']} | {rank_str} | {q['top_similarity']:.4f} | {q['hit_at_1']} | {q['recall_at_1']:.2f} | {q['rr']:.2f} | `{q['matched_chunk_id']}` |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    results = evaluate_retrieval()
    print(format_retrieval_report(results))
