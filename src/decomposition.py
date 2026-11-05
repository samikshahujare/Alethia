"""Question decomposition module for Alethia.

Analyzes composite or multi-intent questions and decomposes them into
independent sub-questions to perform targeted evidence retrieval.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set

from src.llm import call_llm

MAX_SUBQUESTIONS = 3

# Fast-detection regex for coordinating conjunctions followed by interrogative clauses
CONJUNCTION_INTERROGATIVE_PATTERN = re.compile(
    r"(?:,\s*and\s+(?:also\s+)?|;\s*and\s+|,?\s*as well as\s+|,?\s*additionally\s+|,?\s*also\s+|,?\s*and\s+)"
    r"(?:what|which|where|when|why|how|who|whom|whose|explain|describe|list|state|define|name)\b",
    re.IGNORECASE,
)

# Pattern used to split coordinated clauses
CLAUSE_SPLIT_PATTERN = re.compile(
    r"(?:,\s*and\s+(?:also\s+)?|;\s*and\s+|,?\s*as well as\s+|,?\s*additionally\s+|,?\s*also\s+|,?\s*and\s+)"
    r"(?=(?:what|which|where|when|why|how|who|whom|whose|explain|describe|list|state|define|name)\b)",
    re.IGNORECASE,
)

DECOMPOSITION_SYSTEM_PROMPT = (
    "You are an expert NLP question decomposition engine for Retrieval-Augmented Generation.\n"
    "Your task is to analyze the user's composite or multi-intent question and decompose it into "
    "2 or 3 independent, self-contained sub-questions.\n\n"
    "Rules:\n"
    "1. Each sub-question must be standalone, grammatically complete, and directly answerable.\n"
    "2. Resolve all pronouns and coreferences so every sub-question specifies the target entities "
    "explicitly (e.g., 'its beta values' -> 'Adam's beta values').\n"
    "3. Strictly preserve all technical terminology, model names, equations, variable names, and numerical values.\n"
    "4. Never invent new questions, external topics, or facts not requested in the original question.\n"
    "5. Return a maximum of 3 sub-questions. Do not over-decompose.\n"
    "6. Return your output strictly as a JSON object in this exact schema:\n"
    "{\n"
    '  "sub_questions": [\n'
    '    "sub-question 1",\n'
    '    "sub-question 2"\n'
    "  ]\n"
    "}"
)


def is_multi_intent_question(query: Optional[str]) -> bool:
    """Determine whether a query contains multiple independent information requirements.

    Single-intent queries (e.g., 'What is backpropagation?' or 'Define precision and recall')
    return False so they bypass LLM calls and avoid unnecessary decomposition latency.

    Args:
        query: User query string.

    Returns:
        True if the query represents multiple independent intents, False otherwise.
    """
    if not query or not isinstance(query, str):
        return False

    clean_q = query.strip()
    if not clean_q:
        return False

    if len(re.findall(r"\?", clean_q)) >= 2:
        return True

    if CONJUNCTION_INTERROGATIVE_PATTERN.search(clean_q):
        return True

    return False


def decompose_question_rule_based(query: str) -> List[str]:
    """Decompose a composite question into sub-questions using syntax and clause rules.

    Args:
        query: User query string.

    Returns:
        List of 1 to 3 sub-question strings.
    """
    if not query or not isinstance(query, str):
        return []

    clean_q = query.strip()
    if not clean_q:
        return []

    if len(re.findall(r"\?", clean_q)) >= 2:
        parts = [p.strip() for p in re.split(r"\?+", clean_q) if p.strip()]
        sub_questions: List[str] = []
        for p in parts:
            p_clean = p[0].upper() + p[1:] if len(p) > 1 else p.upper()
            if not p_clean.endswith("?"):
                p_clean += "?"
            sub_questions.append(p_clean)

        if len(sub_questions) >= 2:
            return sub_questions[:MAX_SUBQUESTIONS]

    parts = [p.strip() for p in CLAUSE_SPLIT_PATTERN.split(clean_q) if p.strip()]
    if len(parts) >= 2:
        sub_questions = []
        for p in parts:
            p_clean = p.rstrip(",;. ")
            if not p_clean:
                continue
            p_clean = p_clean[0].upper() + p_clean[1:] if len(p_clean) > 1 else p_clean.upper()
            if not p_clean.endswith("?"):
                p_clean += "?"
            sub_questions.append(p_clean)

        if len(sub_questions) >= 2:
            return sub_questions[:MAX_SUBQUESTIONS]

    return [clean_q]


def decompose_question_llm(
    query: str,
    llm_func: Optional[Callable[..., str]] = None,
) -> List[str]:
    """Decompose a composite question into independent sub-questions using LLM assistance.

    Args:
        query: User query string.
        llm_func: Optional callable for LLM invocation (defaults to call_llm).

    Returns:
        List of decomposed sub-questions, guaranteed to be non-empty and bounded.
    """
    if not query or not isinstance(query, str):
        return []

    clean_q = query.strip()
    if not clean_q:
        return []

    caller = llm_func or call_llm
    messages = [
        {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Composite Question: {clean_q}\n\nDecompose into independent sub-questions as strict JSON:"},
    ]

    try:
        raw_response = caller(
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception:
        return decompose_question_rule_based(clean_q)

    if not raw_response or not raw_response.strip():
        return decompose_question_rule_based(clean_q)

    json_text = raw_response.strip()
    match_code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_text, re.DOTALL)
    if match_code_block:
        json_text = match_code_block.group(1)
    else:
        match_brace = re.search(r"\{.*?\}", json_text, re.DOTALL)
        if match_brace:
            json_text = match_brace.group(0)

    try:
        data = json.loads(json_text)
    except Exception:
        return decompose_question_rule_based(clean_q)

    if not isinstance(data, dict):
        return decompose_question_rule_based(clean_q)

    raw_subs = data.get("sub_questions", [])
    if not isinstance(raw_subs, list) or not raw_subs:
        return decompose_question_rule_based(clean_q)

    seen: Set[str] = set()
    cleaned_subs: List[str] = []
    for item in raw_subs:
        if not isinstance(item, str):
            continue
        item_clean = item.strip()
        if not item_clean:
            continue
        normalized = item_clean.lower()
        if normalized not in seen:
            seen.add(normalized)
            cleaned_subs.append(item_clean)

    if not cleaned_subs:
        return decompose_question_rule_based(clean_q)

    return cleaned_subs[:MAX_SUBQUESTIONS]


def decompose_question(
    query: str,
    use_llm_fallback: bool = True,
    llm_func: Optional[Callable[..., str]] = None,
) -> List[str]:
    """Unified entry point for question decomposition.

    First checks if the query is multi-intent. Single-intent queries bypass
    decomposition entirely (0 API calls). Multi-intent queries attempt
    LLM decomposition or rule-based decomposition with safe fallbacks.

    Args:
        query: User query string.
        use_llm_fallback: Whether to attempt LLM decomposition for multi-part questions.
        llm_func: Optional callable for LLM invocation.

    Returns:
        List of 1 to 3 sub-question strings.
    """
    if not query or not isinstance(query, str):
        return []

    clean_q = query.strip()
    if not clean_q:
        return []

    if not is_multi_intent_question(clean_q):
        return [clean_q]

    if use_llm_fallback:
        try:
            llm_subs = decompose_question_llm(clean_q, llm_func=llm_func)
            if len(llm_subs) >= 2:
                return llm_subs
        except Exception:
            pass

    rule_subs = decompose_question_rule_based(clean_q)
    if len(rule_subs) >= 2:
        return rule_subs

    return [clean_q]


def merge_subquestion_chunks(
    subquestion_results: Dict[str, List[Dict[str, Any]]],
    k: int = 4,
) -> List[Dict[str, Any]]:
    """Merge and deduplicate retrieved chunks across multiple sub-questions.

    Combines candidate sets by chunk_id, aggregates matched_subqueries metadata,
    preserves diagnostic rankings, and sorts deterministically by combined RRF score.

    Args:
        subquestion_results: Mapping of sub-question to retrieved chunk list.
        k: Maximum number of merged chunks to return (default: 4).

    Returns:
        List of up to k deduplicated chunk dictionaries.
    """
    candidates: Dict[str, Dict[str, Any]] = {}

    for sq, chunks in subquestion_results.items():
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue

            chunk_copy = dict(chunk)
            existing_meta = dict(chunk_copy.get("metadata", {}))

            if cid not in candidates:
                matched_subs = list(existing_meta.get("matched_subqueries", []))
                if sq not in matched_subs:
                    matched_subs.append(sq)
                existing_meta["matched_subqueries"] = matched_subs
                chunk_copy["metadata"] = existing_meta
                candidates[cid] = chunk_copy
            else:
                target = candidates[cid]
                target_meta = dict(target.get("metadata", {}))
                matched_subs = list(target_meta.get("matched_subqueries", []))
                if sq not in matched_subs:
                    matched_subs.append(sq)
                target_meta["matched_subqueries"] = matched_subs
                target["metadata"] = target_meta

                current_score = float(target.get("rrf_score") or 0.0)
                new_score = float(chunk.get("rrf_score") or 0.0)
                target["rrf_score"] = round(max(current_score, new_score), 6)

    # When multiple sub-questions exist, perform fair round-robin selection
    # across sub-questions so that no sub-question is starved of evidence.
    if len(subquestion_results) > 1 and k > 0:
        selected_ids: List[str] = []
        selected_set: Set[str] = set()

        subq_ordered_ids = {
            sq: [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
            for sq, chunks in subquestion_results.items()
        }

        max_depth = max((len(ids) for ids in subq_ordered_ids.values()), default=0)
        for depth in range(max_depth):
            for sq, ids in subq_ordered_ids.items():
                if depth < len(ids):
                    cid = ids[depth]
                    if cid not in selected_set and cid in candidates:
                        selected_set.add(cid)
                        selected_ids.append(cid)
                        if len(selected_ids) >= k:
                            break
            if len(selected_ids) >= k:
                break

        return [candidates[cid] for cid in selected_ids]

    sorted_chunks = sorted(
        candidates.values(),
        key=lambda c: (-float(c.get("rrf_score") or 0.0), str(c.get("chunk_id", ""))),
    )

    return sorted_chunks[:k]


def _call_retriever(retriever: Any, query: str, k: int, strategy: str) -> List[Dict[str, Any]]:
    """Helper to dispatch retrieval query across diverse retriever interfaces."""
    if hasattr(retriever, "retrieve_hybrid") and strategy == "hybrid":
        return retriever.retrieve_hybrid(query, k=k)
    elif hasattr(retriever, "retrieve_dense") and strategy == "dense":
        return retriever.retrieve_dense(query, k=k)
    elif hasattr(retriever, "retrieve_bm25") and strategy == "bm25":
        return retriever.retrieve_bm25(query, k=k)
    elif hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, k=k, strategy=strategy)
    elif callable(retriever):
        return retriever(query, k=k)
    return []


def retrieve_decomposed(
    query: str,
    k: int = 4,
    strategy: str = "hybrid",
    retriever: Optional[Any] = None,
    llm_func: Optional[Callable[..., str]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Retrieve evidence for a query by intelligently decomposing composite questions.

    Args:
        query: User query string.
        k: Maximum number of final chunks to return.
        strategy: Retrieval strategy ('hybrid', 'dense', or 'bm25').
        retriever: Optional retriever instance or callable. If None, uses get_hybrid_retriever().
        llm_func: Optional LLM callable for decomposition.
        use_llm: Whether to allow LLM assistance for decomposition.

    Returns:
        Dictionary with original_question, decomposed_questions, chunks,
        subquestion_results, retrieval_strategy, and retrieval_diagnostics.
    """
    clean_q = query.strip() if query else ""
    if not clean_q:
        return {
            "original_question": "",
            "decomposed_questions": [],
            "chunks": [],
            "subquestion_results": {},
            "retrieval_strategy": strategy,
            "retrieval_diagnostics": {
                "subquery_count": 0,
                "total_chunks_retrieved": 0,
                "unique_chunks_count": 0,
                "decomposed": False,
            },
        }

    if retriever is None:
        from src.hybrid_retriever import get_hybrid_retriever
        retriever = get_hybrid_retriever()

    sub_questions = decompose_question(clean_q, use_llm_fallback=use_llm, llm_func=llm_func)

    if len(sub_questions) <= 1:
        raw_chunks = _call_retriever(retriever, clean_q, k, strategy)
        for c in raw_chunks:
            meta = dict(c.get("metadata", {}))
            meta["matched_subqueries"] = [clean_q]
            c["metadata"] = meta

        return {
            "original_question": clean_q,
            "decomposed_questions": sub_questions,
            "chunks": raw_chunks,
            "subquestion_results": {clean_q: raw_chunks},
            "retrieval_strategy": strategy,
            "retrieval_diagnostics": {
                "subquery_count": 1,
                "total_chunks_retrieved": len(raw_chunks),
                "unique_chunks_count": len(raw_chunks),
                "decomposed": False,
            },
        }

    sub_results: Dict[str, List[Dict[str, Any]]] = {}
    for sq in sub_questions:
        sq_chunks = _call_retriever(retriever, sq, k, strategy)
        sub_results[sq] = sq_chunks

    merged_chunks = merge_subquestion_chunks(sub_results, k=k)

    return {
        "original_question": clean_q,
        "decomposed_questions": sub_questions,
        "chunks": merged_chunks,
        "subquestion_results": sub_results,
        "retrieval_strategy": strategy,
        "retrieval_diagnostics": {
            "subquery_count": len(sub_questions),
            "total_chunks_retrieved": sum(len(v) for v in sub_results.values()),
            "unique_chunks_count": len(merged_chunks),
            "decomposed": True,
        },
    }

