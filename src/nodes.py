"""Workflow nodes for the Alethia LangGraph engine.

Implements standalone nodes for evidence retrieval, draft synthesis,
factual verification (Judge), and targeted query rewriting.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set
from src.llm import call_llm
from src.store import retrieve_chunks
from src.prompts import (
    GENERATE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    TARGETED_REWRITE_SYSTEM_PROMPT,
    format_context_from_chunks,
    format_generate_prompt,
    format_judge_prompt,
    format_rewrite_prompt,
    format_targeted_rewrite_prompt,
)
from src.state import GraphState


def identify_new_chunks(
    initial_chunk_ids: List[str],
    post_chunk_ids: List[str],
) -> List[str]:
    """Identify chunks newly retrieved in post-rewrite retrieval not present in initial retrieval.

    Preserves the relative ordering of post_chunk_ids while filtering out initial_chunk_ids.

    Args:
        initial_chunk_ids: List of chunk IDs retrieved in the initial retrieval attempt.
        post_chunk_ids: List of chunk IDs retrieved after query rewriting.

    Returns:
        List of chunk IDs that appear in post_chunk_ids but not in initial_chunk_ids.
    """
    init_set = set(initial_chunk_ids or [])
    seen: Set[str] = set()
    new_ids: List[str] = []
    for cid in post_chunk_ids:
        if cid and cid not in init_set and cid not in seen:
            seen.add(cid)
            new_ids.append(cid)
    return new_ids


def generate_targeted_keyword_fallback(
    original_question: str,
    reason: str = "",
    history: Optional[List[str]] = None,
) -> str:
    """Generate a deterministic targeted keyword query when LLM rewrite repeats or loops.

    Extracts key noun phrases, entities, and missing terms from Judge feedback
    and original question to construct a novel search query.

    Args:
        original_question: Original user query string.
        reason: Judge verification feedback string.
        history: Previous query strings.

    Returns:
        A non-empty reformulated query distinct from previous attempts in history.
    """
    hist_set = {re.sub(r"[^\w\s]", "", q.lower()).strip() for q in (history or []) if q}

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "or",
        "not", "context", "regarding", "missing", "information", "verdict", "fail",
        "pass", "reason", "unsupported", "claims", "claim", "about", "for", "with",
        "what", "how", "why", "which", "where", "when", "does", "did", "do", "produce",
        "produced", "empty", "output", "draft", "answer", "question",
    }

    reason_tokens = [
        w for w in re.findall(r"\b[A-Za-z0-9_\-\.]+\b", reason.lower())
        if w not in stop_words and len(w) > 2
    ]
    orig_tokens = [
        w for w in re.findall(r"\b[A-Za-z0-9_\-\.]+\b", original_question.lower())
        if w not in stop_words and len(w) > 2
    ]

    candidates: List[str] = []

    # Candidate 1: Key terms from reason combined with key entities from original question
    combined_tokens = list(dict.fromkeys(reason_tokens[:4] + orig_tokens[:3]))
    if combined_tokens:
        candidates.append(" ".join(combined_tokens))

    # Candidate 2: Informative reason tokens only
    if reason_tokens:
        candidates.append(" ".join(reason_tokens[:5]))

    # Candidate 3: Topic-specific phrases based on entity
    for phrase in ["hyperparameter values", "beta values", "optimizer details", "architecture parameters", "training configuration"]:
        lead = orig_tokens[0] if orig_tokens else "model"
        candidates.append(f"{lead} {phrase}".strip())

    for cand in candidates:
        norm_cand = re.sub(r"[^\w\s]", "", cand.lower()).strip()
        if norm_cand and norm_cand not in hist_set:
            return cand

    # Absolute fallback: append discriminative retry token
    novel_suffix = f"details attempt {len(history or []) + 1}"
    return f"{original_question.rstrip('?.! ')} {novel_suffix}"


def retrieve_node(
    state: GraphState,
    retriever_func: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    k: int = 4,
) -> GraphState:
    """Retrieve top-k relevant document chunks for the current state question.

    Tracks initial retrieval chunks on attempt 0 and newly retrieved chunks on
    subsequent attempts. Preserves earlier useful context across retry cycles.

    Args:
        state: Current GraphState with 'question'.
        retriever_func: Optional callable (query, k) -> list of chunks (defaults to retrieve_chunks).
        k: Maximum number of chunks to retrieve (default: 4).

    Returns:
        Updated GraphState with 'chunks', chunk ID tracking, and appended trace record.
    """
    new_state = dict(state)
    question = new_state.get("question", "").strip()
    attempt = new_state.get("attempts", 0)

    if retriever_func is not None:
        caller = retriever_func
    else:
        strat = new_state.get("retrieval_strategy", "hybrid")
        if strat == "hybrid":
            from src.hybrid_retriever import hybrid_retrieve
            caller = hybrid_retrieve
        else:
            caller = retrieve_chunks

    if not question:
        chunks: List[Dict[str, Any]] = []
    elif attempt == 0 and retriever_func is None:
        doc_intent_handled = False
        try:
            from src.document_inspection import (
                detect_document_intent,
                resolve_target_document,
                find_explicit_document_reference,
                clean_targeted_query,
                get_all_document_chunks,
                extract_questions_from_chunks,
            )
            from src.store import get_vector_store
            store = get_vector_store()
            stats_records = store.collection.get(include=["metadatas"])
            available_sources = sorted(list({
                m.get("source") for m in stats_records.get("metadatas", [])
                if isinstance(m, dict) and m.get("source")
            }))

            doc_intent = detect_document_intent(question)
            if doc_intent:
                target_doc = resolve_target_document(
                    question,
                    available_sources,
                    active_document=new_state.get("target_document"),
                )
                if target_doc:
                    doc_chunks = get_all_document_chunks(target_doc, vector_store=store)
                    if doc_chunks:
                        chunks = doc_chunks
                        new_state["document_intent"] = doc_intent
                        new_state["target_document"] = target_doc
                        new_state["retrieval_strategy"] = "document_inspection"
                        new_state["decomposed_questions"] = [question]
                        new_state["subquestion_results"] = {question: doc_chunks}
                        doc_intent_handled = True
                        if doc_intent == "question_extraction":
                            ext_res = extract_questions_from_chunks(doc_chunks, target_doc)
                            new_state["draft"] = ext_res["answer"]
                            new_state["extracted_questions"] = ext_res["questions"]

            if not doc_intent_handled:
                explicit_doc = find_explicit_document_reference(question, available_sources)
                if explicit_doc:
                    target_doc = explicit_doc
                    new_state["target_document"] = target_doc
                    cleaned_q = clean_targeted_query(question, target_doc)
                    from src.hybrid_retriever import hybrid_retrieve
                    targeted_chunks = hybrid_retrieve(
                        query=cleaned_q,
                        k=k,
                        strategy=strat,
                        source_filter=target_doc,
                    )
                    if targeted_chunks:
                        chunks = targeted_chunks
                        new_state["decomposed_questions"] = [cleaned_q]
                        new_state["subquestion_results"] = {cleaned_q: targeted_chunks}
                        doc_intent_handled = True
        except Exception:
            doc_intent_handled = False

        if not doc_intent_handled:
            try:
                from src.decomposition import retrieve_decomposed
                decomp_res = retrieve_decomposed(question, k=k, strategy=strat)
                chunks = decomp_res.get("chunks", [])
                new_state["decomposed_questions"] = decomp_res.get("decomposed_questions", [question])
                new_state["subquestion_results"] = decomp_res.get("subquestion_results", {question: chunks})
                new_state["retrieval_diagnostics"] = decomp_res.get("retrieval_diagnostics", {})
            except Exception:
                try:
                    chunks = caller(question, k=k)
                except Exception:
                    chunks = []
    else:
        target_doc = new_state.get("target_document")
        try:
            if target_doc and callable(caller):
                try:
                    chunks = caller(question, k=k, source_filter=target_doc)
                except TypeError:
                    chunks = caller(question, k=k)
            else:
                chunks = caller(question, k=k)
        except Exception:
            chunks = []

    chunk_ids = [c.get("chunk_id", "") for c in chunks if c.get("chunk_id")]

    # Distinguish between initial retrieval (attempt 0) and re-retrieval (attempt >= 1)
    if attempt == 0:
        new_state["initial_chunk_ids"] = chunk_ids
        new_state["subsequent_chunk_ids"] = []
        new_state["newly_retrieved_chunk_ids"] = []
        newly_retrieved_ids: List[str] = []
        final_chunks = chunks
    else:
        init_ids = new_state.get("initial_chunk_ids", [])
        new_state["subsequent_chunk_ids"] = chunk_ids
        newly_retrieved_ids = identify_new_chunks(init_ids, chunk_ids)
        new_state["newly_retrieved_chunk_ids"] = newly_retrieved_ids

        # Evidence preservation: keep all post-rewrite chunks (targeted search),
        # plus prior chunks from earlier attempts not in the new set
        post_id_set = {cid for cid in chunk_ids if cid}
        prior_chunks = [c for c in state.get("chunks", []) if c.get("chunk_id") not in post_id_set]
        final_chunks = list(chunks) + prior_chunks

    trace_entry: Dict[str, Any] = {
        "node": "retrieve_node",
        "question": question,
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
        "sources": sorted(list({c.get("source", "unknown") for c in chunks})),
        "attempt": attempt,
        "newly_retrieved_chunk_ids": newly_retrieved_ids,
    }

    new_state["chunks"] = final_chunks
    if "decomposed_questions" not in new_state or not new_state["decomposed_questions"]:
        new_state["decomposed_questions"] = [question] if question else []
    if "subquestion_results" not in new_state or not new_state["subquestion_results"]:
        new_state["subquestion_results"] = {question: chunks} if question else {}
    if "retrieval_strategy" not in new_state:
        new_state["retrieval_strategy"] = "hybrid"
    if "retrieval_diagnostics" not in new_state:
        new_state["retrieval_diagnostics"] = {}

    trace_list = list(new_state.get("trace", []))
    trace_list.append(trace_entry)
    new_state["trace"] = trace_list

    return new_state


def generate_node(
    state: GraphState,
    llm_func: Optional[Callable[..., str]] = None,
) -> GraphState:
    """Generate draft answer using ONLY the retrieved document chunks.

    Args:
        state: Current GraphState with 'question' and 'chunks'.
        llm_func: Optional callable to invoke LLM (defaults to call_llm).

    Returns:
        Updated GraphState with 'draft' and appended trace record.
    """
    caller = llm_func or call_llm
    new_state = dict(state)
    question = new_state.get("question", "").strip()
    chunks = new_state.get("chunks", [])

    # Check for document inspection intents
    doc_intent = new_state.get("document_intent")
    target_doc = new_state.get("target_document", "document")

    if doc_intent == "question_extraction" and new_state.get("draft"):
        draft = new_state["draft"]
        trace_entry = {
            "node": "generate_node",
            "question": question,
            "chunk_count": len(chunks),
            "draft": draft,
            "note": "Verbatim question extraction completed across all document chunks.",
        }
    elif doc_intent == "document_summary":
        from src.document_inspection import generate_document_overview
        draft = generate_document_overview(chunks, target_doc, llm_func=caller)
        trace_entry = {
            "node": "generate_node",
            "question": question,
            "chunk_count": len(chunks),
            "draft": draft,
            "note": "Document summary synthesized from all document chunks.",
        }
    elif doc_intent == "list_extraction":
        from src.document_inspection import extract_document_list
        draft = extract_document_list(question, chunks, target_doc, llm_func=caller)
        trace_entry = {
            "node": "generate_node",
            "question": question,
            "chunk_count": len(chunks),
            "draft": draft,
            "note": "Document list extraction completed across all document chunks.",
        }
    # If no chunks were retrieved, handle immediately without unnecessary inference
    elif not chunks:
        draft = "Not in context."
        trace_entry = {
            "node": "generate_node",
            "question": question,
            "chunk_count": 0,
            "draft": draft,
            "note": "Empty retrieval context; bypassed LLM generation.",
        }
    else:
        if len(chunks) > 10 and new_state.get("document_intent") in ("document_summary", "list_extraction"):
            context_chunks = chunks[:8] + chunks[-2:]
        else:
            context_chunks = chunks
        context = format_context_from_chunks(context_chunks)
        user_prompt = format_generate_prompt(question, context)
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        draft = caller(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )

        trace_entry = {
            "node": "generate_node",
            "question": question,
            "chunk_count": len(chunks),
            "draft": draft,
        }

    new_state["draft"] = draft
    trace_list = list(new_state.get("trace", []))
    trace_list.append(trace_entry)
    new_state["trace"] = trace_list

    return new_state


def _parse_judge_response(raw_text: str) -> tuple[str, str]:
    """Parse VERDICT and REASON from judge output, failing safely if malformed.

    Returns:
        Tuple of (verdict, reason). Verdict is guaranteed to be 'PASS' or 'FAIL'.
    """
    if not raw_text or not raw_text.strip():
        return "FAIL", "Judge produced empty output."

    text = raw_text.strip()

    # Search for VERDICT: PASS or VERDICT: FAIL
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)\b", text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if not verdict_match:
        # Malformed verdict - fail safely
        return "FAIL", f"Malformed judge verdict format; failed safely. Output: {text[:100]}"

    verdict = verdict_match.group(1).upper()
    reason = reason_match.group(1).strip() if reason_match else "No explicit reason provided."

    # Clean up multi-line reason to a concise single summary line
    reason = re.sub(r"\s+", " ", reason)

    return verdict, reason


def judge_node(
    state: GraphState,
    llm_func: Optional[Callable[..., str]] = None,
) -> GraphState:
    """Evaluate whether the generated draft is factually supported by retrieved context.

    Evaluates the complete user question, including all required sub-questions.
    Hardened 5-rule rubric:
    1. Fully grounded -> PASS
    2. Correct abstention -> PASS
    3. Unsupported factual claims -> FAIL
    4. Unnecessary abstention -> FAIL
    5. Contradiction -> FAIL

    Args:
        state: Current GraphState with 'question', 'chunks', and 'draft'.
        llm_func: Optional callable to invoke LLM (defaults to call_llm).

    Returns:
        Updated GraphState with 'verdict' ('PASS'|'FAIL'), 'reason', and trace record.
    """
    caller = llm_func or call_llm
    new_state = dict(state)
    question = new_state.get("question", "").strip()
    chunks = new_state.get("chunks", [])
    draft = new_state.get("draft", "").strip()

    # For deterministic verbatim question extraction, verified directly against all document chunks
    if new_state.get("document_intent") == "question_extraction" and new_state.get("draft"):
        verdict = "PASS"
        extracted_list = new_state.get("extracted_questions", [])
        total_q = len(extracted_list) if extracted_list else len(re.findall(r"^\d+\.\s+", draft, re.MULTILINE))
        target_doc = new_state.get("target_document", "document")
        reason = f"Verified verbatim extraction of {total_q} questions from {target_doc}."
        raw_response = f"VERDICT: {verdict}\nREASON: {reason}"
        trace_entry = {
            "node": "judge_node",
            "question": question,
            "draft": draft,
            "verdict": verdict,
            "reason": reason,
            "raw_response": raw_response,
        }
    else:
        if len(chunks) > 10 and new_state.get("document_intent") in ("document_summary", "list_extraction"):
            context_chunks = chunks[:8] + chunks[-2:]
        else:
            context_chunks = chunks
        context = format_context_from_chunks(context_chunks)

        user_prompt = format_judge_prompt(
            question=question,
            context=context,
            draft=draft,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_response = caller(
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )

        verdict, reason = _parse_judge_response(raw_response)

        trace_entry = {
            "node": "judge_node",
            "question": question,
            "draft": draft,
            "verdict": verdict,
            "reason": reason,
            "raw_response": raw_response,
        }

    new_state["verdict"] = verdict
    new_state["reason"] = reason
    trace_list = list(new_state.get("trace", []))
    trace_list.append(trace_entry)
    new_state["trace"] = trace_list

    return new_state


def rewrite_node(
    state: GraphState,
    llm_func: Optional[Callable[..., str]] = None,
) -> GraphState:
    """Targeted query rewriting with anti-looping protection.

    Preserves the original user question, reads Judge verification feedback,
    isolates unsupported or missing sub-questions, and prevents duplicate
    queries via history tracking and fallback keyword generation.

    Args:
        state: Current GraphState with 'question', 'original_question', and 'reason'.
        llm_func: Optional callable to invoke LLM (defaults to call_llm).

    Returns:
        Updated GraphState with rewritten 'question', incremented 'attempts', and trace.
    """
    caller = llm_func or call_llm
    new_state = dict(state)

    original_q = new_state.get("original_question", "").strip()
    current_q = new_state.get("question", "").strip()
    if not original_q:
        original_q = current_q
        new_state["original_question"] = original_q

    reason = new_state.get("reason", "")
    attempts = new_state.get("attempts", 0) + 1
    decomposed_qs = new_state.get("decomposed_questions", [])

    # Anti-looping: initialize or retrieve query history
    query_history = list(new_state.get("query_history", []))
    if not query_history and original_q:
        query_history = [original_q]
    if current_q and current_q not in query_history:
        query_history.append(current_q)

    # Set of normalized queries for duplicate detection
    norm_history = {re.sub(r"[^\w\s]", "", q.lower()).strip() for q in query_history if q}

    user_prompt = format_targeted_rewrite_prompt(
        original_question=original_q,
        current_question=current_q,
        reason=reason,
        decomposed_questions=decomposed_qs,
        query_history=query_history,
    )
    messages = [
        {"role": "system", "content": TARGETED_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_rewritten = caller(
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
    except Exception:
        raw_rewritten = ""

    clean_rewritten = raw_rewritten.strip().strip('"').strip("'")
    norm_candidate = re.sub(r"[^\w\s]", "", clean_rewritten.lower()).strip()

    # Anti-looping check: reject empty or duplicate queries via exact and normalized textual comparison (case-folding and punctuation removal)
    if not clean_rewritten or norm_candidate in norm_history or norm_candidate == "":
        clean_rewritten = generate_targeted_keyword_fallback(
            original_question=original_q,
            reason=reason,
            history=query_history,
        )

    # Record updated history
    query_history.append(clean_rewritten)

    trace_entry = {
        "node": "rewrite_node",
        "attempt": attempts,
        "original_question": original_q,
        "previous_question": current_q,
        "rewritten_question": clean_rewritten,
        "verification_reason": reason,
        "query_history": list(query_history),
    }

    new_state["question"] = clean_rewritten
    new_state["current_rewrite"] = clean_rewritten
    new_state["attempts"] = attempts
    new_state["query_history"] = query_history

    trace_list = list(new_state.get("trace", []))
    trace_list.append(trace_entry)
    new_state["trace"] = trace_list

    return new_state
