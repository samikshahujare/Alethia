"""Prompt templates and system instructions for Alethia LLM nodes.

Centralizes all prompt strings, formatting routines, and parsing templates
used by generation, judging, and rewriting nodes.
"""

from typing import Any, Dict, List, Optional

# System instructions specified in Phase 3 & Phase 4 requirements
GENERATE_SYSTEM_PROMPT = (
    "You are a strict, factual assistant. Answer using ONLY the provided context.\n"
    "Guidelines:\n"
    "1. Answer using ONLY facts directly stated in the context.\n"
    "2. If a multi-part question has some parts answered in the context and some parts missing, "
    "answer the supported parts fully and explicitly state that the missing parts are not in the context.\n"
    "3. If a term, abbreviation, or concept is defined or explained in the context, provide that definition or explanation as stated. "
    "If it is only mentioned contextually (such as in candidate experience, projects, or activities) but no definition or explanation "
    "is given, state what the context mentions about it and note that the document does not define the abbreviation or term.\n"
    "4. If the topic is completely absent from the context, reply 'Not in context.'\n"
    "5. Never use external knowledge, extrapolate, or invent facts."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker evaluating whether a Generated Answer correctly "
    "and faithfully addresses the entire User Question based ONLY on the retrieved Context.\n"
    "Evaluate the complete question as a whole, including all sub-questions or required parts.\n\n"
    "Evaluation Rules:\n"
    "1. FULLY GROUNDED: If all information required to answer the question is present in the Context "
    "and the Generated Answer is completely supported by the Context without external claims, reply VERDICT: PASS.\n"
    "2. CORRECT ABSTENTION: If information required to answer the question (or any required part) is missing "
    "from the Context and the Generated Answer correctly abstains (e.g., replies 'Not in context.' or states that "
    "the missing information is not in the context), reply VERDICT: PASS.\n"
    "3. UNSUPPORTED CLAIMS: If information required to answer the question is missing from the Context but the "
    "Generated Answer invents or asserts unsupported claims not directly verified by the Context, reply VERDICT: FAIL.\n"
    "4. UNNECESSARY ABSTENTION: If all information required to answer the question is clearly present in the "
    "Context but the Generated Answer unnecessarily abstains (e.g., claims 'Not in context.'), reply VERDICT: FAIL.\n"
    "5. CONTRADICTION: If the Generated Answer contradicts facts directly stated in the Context, reply VERDICT: FAIL.\n\n"
    "Never approve an answer merely because it sounds plausible or authoritative.\n\n"
    "Reply strictly in this format:\n"
    "VERDICT: PASS\n"
    "REASON: <one sentence explaining the factual basis or failure>"
)

REWRITE_SYSTEM_PROMPT = (
    "You are an expert query reformulator. "
    "Rewrite the question to be more specific and improve semantic retrieval relevance. "
    "Return ONLY the question. Do not include introductory or explanatory text."
)

TARGETED_REWRITE_SYSTEM_PROMPT = (
    "You are an expert search query reformulator specializing in RAG systems.\n"
    "The previous retrieval failed to provide sufficient evidence to answer the user's question, "
    "or the generated draft failed factual verification.\n\n"
    "Instructions:\n"
    "1. Analyze the Original Question and the Verification Feedback.\n"
    "2. Identify the specific missing or unsupported information identified by the Judge.\n"
    "3. If the original question contains multiple sub-questions, focus your rewritten query "
    "STRICTLY on the missing or unsupported part. Do NOT re-ask parts that are already answered.\n"
    "4. Preserve all technical terms, model names, equations, variable names, and numerical values.\n"
    "5. Do NOT repeat any previous query from the Query History.\n"
    "6. Return ONLY the single rewritten search query string. Do not include commentary, explanations, or quotes."
)


def format_context_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format a list of retrieved chunk dictionaries into a clean separated context block."""
    if not chunks:
        return ""

    formatted_parts: List[str] = []
    for idx, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        chunk_id = c.get("chunk_id", f"chunk_{idx}")
        text = c.get("text", "").strip()
        formatted_parts.append(f"[Document Chunk {idx} | Source: {source} | ID: {chunk_id}]\n{text}")

    return "\n\n".join(formatted_parts)


def format_generate_prompt(question: str, context: str) -> str:
    """Format user prompt for the Generate node."""
    if not context.strip():
        context = "(No retrieved context available)"

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def format_judge_prompt(question: str, context: str, draft: str) -> str:
    """Format user prompt for the Judge node."""
    if not context.strip():
        context = "(Empty context)"

    return (
        f"Retrieved Context:\n{context}\n\n"
        f"User Question:\n{question}\n\n"
        f"Generated Answer:\n{draft}\n\n"
        f"Evaluate whether the Generated Answer correctly and faithfully handles the entire User Question against the Retrieved Context."
    )


def format_rewrite_prompt(original_question: str, current_question: str, reason: str = "") -> str:
    """Format user prompt for the Rewrite node."""
    parts = [f"Original Question: {original_question}"]
    if current_question and current_question != original_question:
        parts.append(f"Current Query Attempt: {current_question}")
    if reason:
        parts.append(f"Verification Feedback: {reason}")
    parts.append("Rewritten Question:")

    return "\n\n".join(parts)


def format_targeted_rewrite_prompt(
    original_question: str,
    current_question: str,
    reason: str = "",
    decomposed_questions: Optional[List[str]] = None,
    query_history: Optional[List[str]] = None,
) -> str:
    """Format structured user prompt for targeted query reformulation."""
    parts = [f"Original User Question: {original_question}"]

    if decomposed_questions and len(decomposed_questions) > 1:
        sub_list = "\n".join(f"- {sq}" for sq in decomposed_questions)
        parts.append(f"Decomposed Information Requirements:\n{sub_list}")

    if query_history and len(query_history) > 1:
        hist_list = ", ".join(f'"{q}"' for q in query_history)
        parts.append(f"Previous Attempted Queries (Do NOT repeat): {hist_list}")
    elif current_question and current_question != original_question:
        parts.append(f"Previous Query Attempt: {current_question}")

    if reason:
        parts.append(f"Judge Verification Feedback: {reason}")

    parts.append(
        "Generate a targeted search query addressing ONLY the missing/unsupported requirement:"
    )

    return "\n\n".join(parts)
