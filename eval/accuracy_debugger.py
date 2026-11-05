"""Systematic Answer Accuracy Debugger for Alethia.

Traces and records all 18 diagnostic execution fields across the RAG pipeline
and categorizes failures into 10 structured failure categories:
1. Document ingestion failure
2. Retrieval failure
3. Wrong document selection
4. Context construction failure
5. Question decomposition failure
6. Unsupported generation
7. Incorrect Judge approval
8. Unnecessary abstention
9. Incomplete answer
10. Source attribution failure
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set

from src.bm25 import BM25Index
from src.decomposition import decompose_question, retrieve_decomposed
from src.document_inspection import (
    detect_document_intent,
    extract_questions_from_chunks,
    get_all_document_chunks,
    resolve_target_document,
)
from src.graph import run_rag_pipeline as run_workflow
from src.hybrid_retriever import hybrid_retrieve
from src.prompts import format_context_from_chunks
from src.retriever import retrieve_chunks as retrieve_relevant_chunks
from src.store import get_vector_store


# Failure category definitions
FAILURE_CATEGORIES = [
    "document_ingestion_failure",
    "retrieval_failure",
    "wrong_document_selection",
    "context_construction_failure",
    "question_decomposition_failure",
    "unsupported_generation",
    "incorrect_judge_approval",
    "unnecessary_abstention",
    "incomplete_answer",
    "source_attribution_failure",
]


@dataclass
class DiagnosticRecord:
    """Complete 18-field diagnostic trace for an Alethia query execution."""
    # 1. Original user question
    original_question: str = ""
    # 2. Detected query intent
    detected_intent: Optional[str] = None
    # 3. Selected document
    selected_document: Optional[str] = None
    # 4. Decomposed subquestions
    decomposed_subquestions: List[str] = field(default_factory=list)
    # 5. Dense retrieval results (list of {chunk_id, score, text_snippet})
    dense_results: List[Dict[str, Any]] = field(default_factory=list)
    # 6. BM25 retrieval results (list of {chunk_id, score, text_snippet})
    bm25_results: List[Dict[str, Any]] = field(default_factory=list)
    # 7. Hybrid retrieval results (list of {chunk_id, rrf_score, text_snippet})
    hybrid_results: List[Dict[str, Any]] = field(default_factory=list)
    # 8. RRF rankings (chunk_id -> {dense_rank, bm25_rank, rrf_score})
    rrf_rankings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 9. Retrieved chunk IDs
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    # 10. Actual retrieved text (chunk_id -> text)
    actual_retrieved_text: Dict[str, str] = field(default_factory=dict)
    # 11. Context passed to generator
    context_passed_to_generator: str = ""
    # 12. Generated answer (first draft)
    generated_answer: str = ""
    # 13. Judge verdict and reason
    judge_verdict: str = ""
    judge_reason: str = ""
    # 14. Rewrite attempts
    rewrite_attempts: int = 0
    # 15. Newly retrieved evidence
    newly_retrieved_evidence: List[str] = field(default_factory=list)
    # 16. Final answer
    final_answer: str = ""
    # 17. Source references (sources cited or attributed)
    source_references: List[str] = field(default_factory=list)
    # 18. Total execution time in ms
    total_execution_time_ms: float = 0.0

    # Diagnostic metadata
    failure_category: Optional[str] = None
    diagnostic_notes: List[str] = field(default_factory=list)
    is_correct: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnostic record to dictionary."""
        return asdict(self)


def diagnose_query(
    question: str,
    expected_answer_keywords: Optional[List[str]] = None,
    expected_chunks: Optional[List[str]] = None,
    expected_abstain: bool = False,
    expected_source: Optional[str] = None,
    k: int = 4,
    llm_func: Optional[Callable[..., str]] = None,
) -> DiagnosticRecord:
    """Execute end-to-end diagnosis of a question through Alethia recording all 18 fields.

    Args:
        question: User query string.
        expected_answer_keywords: Optional list of keywords expected in correct answer.
        expected_chunks: Optional list of chunk IDs expected to be retrieved.
        expected_abstain: Whether the correct response is abstention ('Not in context').
        expected_source: Expected document name.
        k: Retrieval chunk limit.
        llm_func: Optional LLM callable override for deterministic testing.

    Returns:
        Populated DiagnosticRecord with full 18 fields and categorized failure.
    """
    record = DiagnosticRecord(original_question=question)
    start_time = time.perf_counter()

    store = get_vector_store()
    available_metas = store.collection.get(include=["metadatas"]).get("metadatas") or []
    available_sources = sorted(list({m.get("source", "") for m in available_metas if m.get("source")}))

    # 1 & 2. Query Intent Detection
    intent = detect_document_intent(question)
    record.detected_intent = intent

    # 3. Selected Document
    selected_doc = resolve_target_document(question, available_sources)
    record.selected_document = selected_doc

    # Check wrong document selection if expectation provided
    if expected_source and selected_doc and selected_doc != expected_source:
        record.diagnostic_notes.append(
            f"Selected document '{selected_doc}' differs from expected '{expected_source}'."
        )

    # 4. Question Decomposition
    sub_questions = decompose_question(question)
    record.decomposed_subquestions = sub_questions

    # 5. Dense Retrieval
    try:
        dense_chunks = retrieve_relevant_chunks(question, k=max(k * 3, 10))
        record.dense_results = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "score": c.get("similarity_score", 0.0),
                "source": c.get("source", ""),
                "text_snippet": c.get("text", "")[:120],
            }
            for c in dense_chunks[:k]
        ]
    except Exception as e:
        record.diagnostic_notes.append(f"Dense retrieval error: {str(e)}")

    # 6. BM25 Retrieval
    try:
        from src.hybrid_retriever import _get_or_build_bm25
        bm25_index = _get_or_build_bm25()
        bm25_matches = bm25_index.search(question, top_k=max(k * 3, 10))
        record.bm25_results = [
            {
                "chunk_id": m.get("chunk_id", ""),
                "score": m.get("bm25_score", 0.0),
                "source": m.get("source", ""),
                "text_snippet": m.get("text", "")[:120],
            }
            for m in bm25_matches[:k]
        ]
    except Exception as e:
        record.diagnostic_notes.append(f"BM25 retrieval error: {str(e)}")

    # 7 & 8. Hybrid Retrieval and RRF Rankings
    try:
        hybrid_chunks = hybrid_retrieve(question, k=k)
        record.hybrid_results = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "rrf_score": c.get("rrf_score", 0.0),
                "source": c.get("source", ""),
                "text_snippet": c.get("text", "")[:120],
            }
            for c in hybrid_chunks
        ]
        for c in hybrid_chunks:
            cid = c.get("chunk_id", "")
            if cid:
                record.rrf_rankings[cid] = {
                    "dense_rank": c.get("dense_rank"),
                    "bm25_rank": c.get("bm25_rank"),
                    "rrf_score": c.get("rrf_score"),
                }
    except Exception as e:
        record.diagnostic_notes.append(f"Hybrid retrieval error: {str(e)}")

    # 9 & 10. Run Full Workflow to capture state transitions, generation, and judge
    try:
        workflow_res = run_workflow(question, llm_func=llm_func, k=k)
        final_chunks = workflow_res.get("chunks", [])

        record.retrieved_chunk_ids = [c.get("chunk_id", "") for c in final_chunks if c.get("chunk_id")]
        record.actual_retrieved_text = {
            c.get("chunk_id", ""): c.get("text", "")
            for c in final_chunks
            if c.get("chunk_id")
        }

        # 11. Context passed to generator
        record.context_passed_to_generator = format_context_from_chunks(final_chunks)

        # 12. Generated answer & 16. Final answer
        record.final_answer = workflow_res.get("final_answer", "") or workflow_res.get("draft", "")

        # Trace extraction for first draft, judge reason, rewrites, new evidence
        trace = workflow_res.get("trace", [])
        drafts = [e.get("draft") for e in trace if e.get("node") == "generate_node" and e.get("draft")]
        record.generated_answer = drafts[0] if drafts else record.final_answer

        judge_events = [e for e in trace if e.get("node") == "judge_node"]
        if judge_events:
            last_judge = judge_events[-1]
            record.judge_verdict = last_judge.get("verdict", "")
            record.judge_reason = last_judge.get("reason", "")
        else:
            record.judge_verdict = workflow_res.get("verdict", "")
            record.judge_reason = workflow_res.get("reason", "")

        # 14. Rewrite attempts
        record.rewrite_attempts = workflow_res.get("rewrite_attempts", 0)

        # 15. Newly retrieved evidence
        record.newly_retrieved_evidence = workflow_res.get("newly_retrieved_chunk_ids", [])

        # 17. Source references
        sources = sorted(list({c.get("source", "") for c in final_chunks if c.get("source")}))
        record.source_references = sources

    except Exception as e:
        record.diagnostic_notes.append(f"Workflow execution error: {str(e)}")

    # 18. Total execution time
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    record.total_execution_time_ms = round(elapsed_ms, 2)

    # Automatic Failure Classification
    _classify_failure(
        record,
        expected_answer_keywords=expected_answer_keywords,
        expected_chunks=expected_chunks,
        expected_abstain=expected_abstain,
        expected_source=expected_source,
    )

    return record


def _classify_failure(
    record: DiagnosticRecord,
    expected_answer_keywords: Optional[List[str]] = None,
    expected_chunks: Optional[List[str]] = None,
    expected_abstain: bool = False,
    expected_source: Optional[str] = None,
) -> None:
    """Classify diagnostic record into one of 10 structured failure categories."""
    final_ans_lower = (record.final_answer or "").strip().lower()
    is_abstained = "not in context" in final_ans_lower

    # 1. Expected Abstention check
    if expected_abstain:
        if is_abstained:
            record.is_correct = True
            record.failure_category = None
            return
        else:
            record.is_correct = False
            record.failure_category = "unsupported_generation"
            record.diagnostic_notes.append("System answered out-of-scope question instead of abstaining.")
            return

    # 2. Wrong document selection check (upstream cause of bad retrieval/abstention)
    if expected_source and record.selected_document and record.selected_document != expected_source:
        record.is_correct = False
        record.failure_category = "wrong_document_selection"
        record.diagnostic_notes.append(
            f"Target document resolved as '{record.selected_document}' instead of '{expected_source}'."
        )
        return

    # 3. Retrieval failure check (expected chunks missing from retrieval)
    if expected_chunks:
        retrieved_set = set(record.retrieved_chunk_ids)
        missing_chunks = [cid for cid in expected_chunks if cid not in retrieved_set]
        if missing_chunks:
            record.is_correct = False
            record.failure_category = "retrieval_failure"
            record.diagnostic_notes.append(f"Expected chunk(s) missing from retrieval: {missing_chunks}")
            return

    # 4. Unnecessary Abstention check (relevant chunks and document were present, but generator abstained)
    if is_abstained and not expected_abstain:
        record.is_correct = False
        record.failure_category = "unnecessary_abstention"
        record.diagnostic_notes.append("System abstained ('Not in context') when factual answer was expected.")
        return

    # 5. Answer Keyword / Content Verification
    if expected_answer_keywords:
        missing_kw = [kw for kw in expected_answer_keywords if kw.lower() not in final_ans_lower]
        if missing_kw:
            record.is_correct = False
            if len(missing_kw) < len(expected_answer_keywords):
                record.failure_category = "incomplete_answer"
                record.diagnostic_notes.append(f"Partial answer; missing keywords: {missing_kw}")
            else:
                record.failure_category = "unsupported_generation"
                record.diagnostic_notes.append(f"Answer failed to contain expected keywords: {missing_kw}")
            return

    # 6. Judge Verification check
    if record.judge_verdict.upper() == "FAIL":
        record.is_correct = False
        record.failure_category = "unsupported_generation"
        record.diagnostic_notes.append(f"Judge failed answer: {record.judge_reason}")
        return

    record.is_correct = True
    record.failure_category = None
