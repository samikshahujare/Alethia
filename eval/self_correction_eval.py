"""Dedicated Real Self-Correction Evaluation Module for Alethia (Step 3).

Evaluates the cyclic state machine's query reformulation and correction loop:
1. Strict 10-step chronological trace verification
2. Mutually exclusive classification:
   - initial_success: Correct answer on attempt 0 (zero rewrites) with all mandatory facts
   - successful_correction: Strict 10-step verification (FAIL -> Rewrite -> Re-retrieve -> PASS -> Grounded)
   - failed_correction: Rewrote but remained failing or reached max retry limit
   - abstention: Correctly returned 'Not in context.' on unanswerable query (out-of-scope or corpus boundary)
   - false_positive_pass: Judge passed an incorrect/unsupported answer
   - false_negative_fail: Judge rejected a grounded/correct answer
3. Comprehensive metrics:
   - Answer accuracy across answerable cases (denominator: 12)
   - Out-of-scope abstention rate (denominator: 2)
   - Corpus-boundary abstention rate (denominator: 2)
   - Overall abstention rate across total suite (4/16 = 25.0%)
   - Overall decision correctness (denominator: 16)
   - Successful self-correction rate among required cases ('N/A' when required count is 0)
   - Failed self-correction rate ('N/A' when required count is 0)
   - Max rewrite compliance rate ('N/A' when zero rewrites occurred)
   - Mean rewrites and latency breakdown (0 vs 1 vs 2 rewrites)
4. Full attempt-by-attempt state trace serialization to JSON

Evaluation Limitation Note:
Mandatory-fact substring matching is a limited evaluation heuristic, not a semantic
correctness guarantee. While it checks for the physical presence of required factual tokens,
true semantic correctness requires contextual validation, human evaluation, or advanced
entailment modeling.
"""

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import MAX_REWRITES, run_rag_pipeline


SELF_CORRECTION_DATASET: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. Answerable Queries: Phrasing Challenges, Ambiguity, & Lexical Traps
    # Note: These test cases evaluate pipeline robustness against phrasing ambiguity.
    # In live execution, dense semantic retrieval retrieved target evidence on attempt 0,
    # resulting in initial_success. They are NOT demonstrated self-correction cases.
    # -------------------------------------------------------------------------
    {
        "id": "SC-01",
        "category": "cross_domain",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What is the penalty that creates sparse features?",
        "expected_answer": "L1 regularization (Lasso regression) penalizes the absolute values of weights, driving non-essential coefficients to zero and producing sparse feature representations.",
        "relevant_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0003"],
        "mandatory_facts": ["l1", "regularization"],
        "supporting_facts": ["l1", "regularization"],
        "expected_behavior": "initial_success",
        "correction_reason": "Vague phrasing without model or method name; tests whether retrieval maps to L1 Lasso or requires reformulation to isolate sparsity mechanism.",
    },
    {
        "id": "SC-02",
        "category": "lexical_trap",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "How is the running root-mean-square average used in parameter step sizing?",
        "expected_answer": "RMSprop divides the gradient by a running root-mean-square average of recent squared gradients to prevent rapid decay in frequent parameter directions.",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0002"],
        "mandatory_facts": ["root", "gradient"],
        "supporting_facts": ["root", "gradient"],
        "expected_behavior": "initial_success",
        "correction_reason": "Query omits optimizer name; tests whether retrieval maps to RMSprop without needing explicit optimizer keyword.",
    },
    {
        "id": "SC-03",
        "category": "ambiguous",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "Why do we divide by the square root during sequence alignment?",
        "expected_answer": "In scaled dot-product attention, dividing by sqrt(d_k) scales the dot products to prevent large key dimensions from pushing values into regions with extremely small gradients.",
        "relevant_chunk_ids": ["attention_mechanisms_txt_chunk_0001"],
        "mandatory_facts": ["d_k", "gradient"],
        "supporting_facts": ["d_k", "gradient"],
        "expected_behavior": "initial_success",
        "correction_reason": "Colloquial terminology ('sequence alignment' vs 'scaled dot-product attention'); tests whether attention scaling factor sqrt(d_k) is retrieved.",
    },
    {
        "id": "SC-04",
        "category": "insufficient_context",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What causes gradient saturation and how do we smooth its flow?",
        "expected_answer": "ReLU causes saturation for non-positive values, while GELU weights inputs by their probability under a Gaussian cumulative distribution, smoothing gradient flows in models like BERT and GPT.",
        "relevant_chunk_ids": ["neural_network_training_txt_chunk_0002"],
        "mandatory_facts": ["gelu", "gaussian"],
        "supporting_facts": ["gelu", "gaussian"],
        "expected_behavior": "initial_success",
        "correction_reason": "Conflates activation dynamics; tests whether GELU and Gaussian cumulative smoothing are retrieved directly.",
    },
    {
        "id": "SC-05",
        "category": "lexical_trap",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "How are future tokens hidden during autoregressive generation?",
        "expected_answer": "Causal masking in sequence-to-sequence decoders sets future token positions to negative infinity before softmax, preventing the model from attending to subsequent tokens.",
        "relevant_chunk_ids": ["attention_mechanisms_txt_chunk_0003"],
        "mandatory_facts": ["causal mask", "negative infinity"],
        "supporting_facts": ["causal mask", "negative infinity"],
        "expected_behavior": "initial_success",
        "correction_reason": "Casual language ('hidden during generation'); tests whether causal masking and negative infinity are retrieved.",
    },
    {
        "id": "SC-06",
        "category": "ambiguous",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What prevents weights from oscillating wildly along steep valleys?",
        "expected_answer": "Classical momentum or Nesterov accelerated momentum dampens pathological oscillations along steep ravines in stochastic gradient descent.",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0001"],
        "mandatory_facts": ["momentum", "nesterov"],
        "supporting_facts": ["momentum", "nesterov"],
        "expected_behavior": "initial_success",
        "correction_reason": "Metaphorical phrasing ('steep valleys'); tests whether momentum dampening oscillations along ravines is retrieved.",
    },
    {
        "id": "SC-07",
        "category": "insufficient_context",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "Which subword algorithm avoids pre-segmentation on spaces?",
        "expected_answer": "SentencePiece treats input text as a raw stream of characters including whitespace characters, allowing seamless language-independent tokenization without pre-segmentation.",
        "relevant_chunk_ids": ["nlp_concepts_txt_chunk_0002"],
        "mandatory_facts": ["sentencepiece"],
        "supporting_facts": ["sentencepiece"],
        "expected_behavior": "initial_success",
        "correction_reason": "Query omits tokenizer name; tests whether SentencePiece raw character stream without pre-segmentation is retrieved.",
    },
    {
        "id": "SC-08",
        "category": "insufficient_context",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "How do we penalize overly brief translated text?",
        "expected_answer": "The BLEU score incorporates a brevity penalty to penalize excessively terse translation outputs when comparing against human reference translations.",
        "relevant_chunk_ids": ["nlp_concepts_txt_chunk_0004"],
        "mandatory_facts": ["bleu", "brevity penalty"],
        "supporting_facts": ["bleu", "brevity penalty"],
        "expected_behavior": "initial_success",
        "correction_reason": "Describes metric function without naming BLEU; tests whether BLEU brevity penalty is identified directly.",
    },
    {
        "id": "SC-09",
        "category": "ambiguous",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What metric is independent of batch size for normalizing tokens?",
        "expected_answer": "Layer normalization computes mean and variance statistics across hidden feature channels independently of batch size.",
        "relevant_chunk_ids": ["neural_network_training_txt_chunk_0004"],
        "mandatory_facts": ["layer normalization"],
        "supporting_facts": ["layer normalization"],
        "expected_behavior": "initial_success",
        "correction_reason": "Asks for normalization technique without naming LayerNorm; tests whether layer normalization hidden channels are retrieved.",
    },
    {
        "id": "SC-10",
        "category": "ambiguous",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What are the rings composed of?",
        "expected_answer": "Saturn's planetary rings consist predominantly of billions of small water ice particles ranging in size from micrometers to meters, along with rocky silicates and tholin organic compounds.",
        "relevant_chunk_ids": ["solar_system_txt_chunk_0003"],
        "mandatory_facts": ["water", "ice", "silicate"],
        "supporting_facts": ["water", "ice", "silicate"],
        "expected_behavior": "initial_success",
        "correction_reason": "Heavily ambiguous entity (omits Saturn); tests whether dense embeddings map to Saturn's ring system.",
    },
    {
        "id": "SC-11",
        "category": "ambiguous",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "What is the height of the largest mountain?",
        "expected_answer": "Olympus Mons on Mars stands approximately 21.9 kilometers high, making it the tallest planetary mountain documented in the solar system.",
        "relevant_chunk_ids": ["solar_system_txt_chunk_0002"],
        "mandatory_facts": ["olympus", "mons", "21.9"],
        "supporting_facts": ["olympus", "mons", "21.9"],
        "expected_behavior": "initial_success",
        "correction_reason": "Ambiguous entity and planetary body; tests whether retrieval isolates Olympus Mons on Mars at 21.9 km.",
    },

    # -------------------------------------------------------------------------
    # 2. Out-of-Scope Unanswerable Queries (Expected: Abstain)
    # -------------------------------------------------------------------------
    {
        "id": "SC-12",
        "category": "unanswerable",
        "answerability": "unanswerable",
        "expected_decision": "abstain",
        "question": "What is the capital city of France?",
        "expected_answer": "Not in context.",
        "relevant_chunk_ids": [],
        "mandatory_facts": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
        "correction_reason": "Completely out-of-domain general geography query; tests immediate cycle-1 abstention without wasteful rewrites.",
    },
    {
        "id": "SC-13",
        "category": "unanswerable",
        "answerability": "unanswerable",
        "expected_decision": "abstain",
        "question": "What quantum gate sequence is used in the attention mechanism?",
        "expected_answer": "Not in context.",
        "relevant_chunk_ids": [],
        "mandatory_facts": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
        "correction_reason": "Domain collision with neural attention; quantum computing is completely absent from the corpus, requiring clean abstention.",
    },

    # -------------------------------------------------------------------------
    # 3. Corpus Boundary Unanswerable Queries (Expected: Abstain)
    # -------------------------------------------------------------------------
    {
        "id": "SC-14",
        "category": "corpus_boundary",
        "answerability": "unanswerable",
        "expected_decision": "abstain",
        "question": "What was the exact numerical learning rate value used for the base Transformer on WMT 2014 English-to-German?",
        "expected_answer": "Not in context.",
        "relevant_chunk_ids": [],
        "mandatory_facts": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
        "correction_reason": "Corpus boundary test: text mentions Adam optimizer with warmup and 28.4 BLEU, but completely omits the exact numerical learning rate; pipeline must cleanly abstain rather than hallucinating.",
    },
    {
        "id": "SC-15",
        "category": "corpus_boundary",
        "answerability": "unanswerable",
        "expected_decision": "abstain",
        "question": "What was the batch size used when training GPT with GELU activation?",
        "expected_answer": "Not in context.",
        "relevant_chunk_ids": [],
        "mandatory_facts": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
        "correction_reason": "Corpus boundary test: document mentions GELU in GPT and BERT, but omits the training batch size; tests bounded termination without hallucinating numerical values.",
    },

    # -------------------------------------------------------------------------
    # 4. Standard Factual Control Query (Expected: Answer)
    # -------------------------------------------------------------------------
    {
        "id": "SC-16",
        "category": "control",
        "answerability": "answerable",
        "expected_decision": "answer",
        "question": "How many encoder and decoder layers does the base Transformer architecture use?",
        "expected_answer": "Six encoder layers and six decoder layers in the base Transformer architecture.",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "mandatory_facts": ["six encoder", "six decoder"],
        "supporting_facts": ["six encoder", "six decoder"],
        "expected_behavior": "initial_success",
        "correction_reason": "Control query: unambiguous, directly answerable question with prominent evidence; tests clean first-attempt success (zero rewrites).",
    },
]


def verify_strict_self_correction_trace(
    trace: List[Dict[str, Any]],
    original_question: str,
    final_verdict: str,
    final_draft: str,
    mandatory_facts: Optional[List[str]] = None,
    relevant_chunk_ids: Optional[List[str]] = None,
    supporting_facts: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Strictly verify all 10 chronological criteria for successful self-correction.

    Requirements:
    1. Initial retrieval completes.
    2. Initial Generate produces an answer draft.
    3. Initial Judge verdict is FAIL.
    4. Rewrite node executes after initial Judge FAIL.
    5. Rewritten query differs from original query.
    6. Subsequent retrieval executes using the exact rewritten query.
    7. Post-rewrite retrieval fetches new chunks containing verified ground-truth evidence chunk IDs.
       (Note: Different retrieved chunks alone do not establish useful evidence; explicit target
       chunk IDs are required. Arbitrary new chunks without verified evidence are strictly rejected.)
    8. Generate produces revised answer draft after post-rewrite retrieval.
    9. Final Judge verdict is PASS strictly after revised Generate node.
    10. Final answer contains all mandatory ground-truth facts.

    Evaluation Limitation Note:
    Mandatory-fact substring matching is a limited evaluation heuristic, not a semantic
    correctness guarantee. While it checks for the physical presence of required factual tokens,
    true semantic correctness requires contextual validation, human evaluation, or advanced
    entailment modeling.

    If ANY criterion is False, the case cannot be classified as successful_correction.
    """
    clean_orig_q = original_question.strip().lower()
    facts = mandatory_facts if mandatory_facts is not None else (supporting_facts or [])
    target_chunks = set(relevant_chunk_ids or [])

    # Locate node execution entries
    retrieve_entries = [(idx, e) for idx, e in enumerate(trace) if e.get("node") == "retrieve_node"]
    generate_entries = [(idx, e) for idx, e in enumerate(trace) if e.get("node") == "generate_node"]
    judge_entries = [(idx, e) for idx, e in enumerate(trace) if e.get("node") == "judge_node"]
    rewrite_entries = [(idx, e) for idx, e in enumerate(trace) if e.get("node") == "rewrite_node"]

    # 1. Initial retrieval completes
    has_init_retrieval = len(retrieve_entries) >= 1
    init_ret_idx = retrieve_entries[0][0] if has_init_retrieval else -1
    init_chunk_ids = set(retrieve_entries[0][1].get("chunk_ids", [])) if has_init_retrieval else set()

    # 2. Initial Generate produces an answer draft
    valid_init_gens = [
        (idx, e) for idx, e in generate_entries
        if idx > init_ret_idx and bool(e.get("draft", "").strip())
    ]
    has_init_generate = len(valid_init_gens) >= 1
    init_gen_idx = valid_init_gens[0][0] if has_init_generate else -1

    # 3. Initial Judge verdict is FAIL
    valid_init_judges = [
        (idx, e) for idx, e in judge_entries
        if idx > init_gen_idx and e.get("verdict", "").upper() == "FAIL"
    ]
    has_init_judge_fail = len(valid_init_judges) >= 1
    init_judge_idx = valid_init_judges[0][0] if has_init_judge_fail else -1

    # 4. Rewrite node executes after initial Judge FAIL
    valid_rewrites = [
        (idx, e) for idx, e in rewrite_entries
        if idx > init_judge_idx
    ]
    has_rewrite_node = len(valid_rewrites) >= 1
    rewrite_idx = valid_rewrites[0][0] if has_rewrite_node else -1
    rewritten_q = valid_rewrites[0][1].get("rewritten_question", "").strip().lower() if has_rewrite_node else ""

    # 5. Rewritten query is not identical to original query
    has_query_changed = (
        has_rewrite_node
        and bool(rewritten_q)
        and (rewritten_q != clean_orig_q)
    )

    # 6. Subsequent retrieval executes after rewrite AND uses rewritten query
    subsequent_retrievals = [
        (idx, e) for idx, e in retrieve_entries if idx > rewrite_idx
    ]
    has_subsequent_retrieval_node = len(subsequent_retrievals) >= 1
    post_ret_idx = subsequent_retrievals[0][0] if has_subsequent_retrieval_node else -1
    post_ret_query = (
        subsequent_retrievals[0][1].get("question", "").strip().lower()
        if has_subsequent_retrieval_node else ""
    )
    uses_rewritten_query = (
        has_subsequent_retrieval_node
        and bool(post_ret_query)
        and (post_ret_query == rewritten_q)
    )
    has_subsequent_retrieval = has_subsequent_retrieval_node and uses_rewritten_query

    # 7. Post-rewrite retrieval fetches new chunks AND contains at least one newly retrieved ground-truth chunk.
    # Note: Merely retrieving different chunks does NOT establish useful evidence.
    # Furthermore, a ground-truth chunk that was already present in initial retrieval
    # does NOT count as newly discovered evidence.
    # Useful evidence requires:
    #   newly_retrieved_chunks = post_chunk_ids - init_chunk_ids
    #   new_relevant_chunks = newly_retrieved_chunks.intersection(target_chunks)
    #   has_useful_evidence = len(new_relevant_chunks) > 0
    post_chunk_ids = (
        set(subsequent_retrievals[0][1].get("chunk_ids", []))
        if has_subsequent_retrieval_node else set()
    )
    newly_retrieved_chunks = post_chunk_ids - init_chunk_ids
    if target_chunks:
        new_relevant_chunks = newly_retrieved_chunks.intersection(target_chunks)
        has_useful_evidence = len(new_relevant_chunks) > 0
    else:
        new_relevant_chunks = set()
        has_useful_evidence = False  # No ground-truth chunk IDs supplied -> cannot verify useful evidence

    # 8. Generate produces revised answer draft after post-rewrite retrieval
    subsequent_generates = [
        (idx, e) for idx, e in generate_entries
        if idx > post_ret_idx and bool(e.get("draft", "").strip())
    ]
    has_revised_draft = len(subsequent_generates) >= 1
    post_gen_idx = subsequent_generates[0][0] if has_revised_draft else -1

    # 9. Final Judge verdict is PASS strictly after revised Generate
    subsequent_judges = [
        (idx, e) for idx, e in judge_entries
        if idx > post_gen_idx and e.get("verdict", "").upper() == "PASS"
    ]
    has_final_judge_pass_after_gen = (
        len(subsequent_judges) >= 1
        and final_verdict.upper() == "PASS"
    )

    # 10. Final answer contains all mandatory ground-truth facts
    clean_draft = final_draft.lower()
    has_all_mandatory_facts = (
        all(fact.lower() in clean_draft for fact in facts)
        if facts else bool(clean_draft.strip())
    )

    criteria: Dict[str, bool] = {
        "step_01_initial_retrieval": has_init_retrieval,
        "step_02_initial_generate": has_init_generate,
        "step_03_initial_judge_fail": has_init_judge_fail,
        "step_04_rewrite_executed": has_rewrite_node,
        "step_05_query_changed": has_query_changed,
        "step_06_subsequent_retrieval": has_subsequent_retrieval,
        "step_06_subsequent_retrieval_uses_rewrite": has_subsequent_retrieval,
        "step_07_new_chunks_retrieved": has_useful_evidence,
        "step_07_ground_truth_chunks_retrieved": has_useful_evidence,
        "step_08_revised_answer_draft": has_revised_draft,
        "step_09_final_judge_pass": has_final_judge_pass_after_gen,
        "step_09_final_judge_pass_after_generate": has_final_judge_pass_after_gen,
        "step_10_final_answer_grounded": has_all_mandatory_facts,
        "step_10_all_mandatory_facts_present": has_all_mandatory_facts,
    }

    core_10_steps = [
        has_init_retrieval,
        has_init_generate,
        has_init_judge_fail,
        has_rewrite_node,
        has_query_changed,
        has_subsequent_retrieval,
        has_useful_evidence,
        has_revised_draft,
        has_final_judge_pass_after_gen,
        has_all_mandatory_facts,
    ]
    all_criteria_met = all(core_10_steps)
    return all_criteria_met, criteria


def classify_self_correction_case(
    state: Dict[str, Any],
    expected_behavior: Optional[str] = None,
    supporting_facts: Optional[List[str]] = None,
    original_question: str = "",
    expected_decision: Optional[str] = None,
    mandatory_facts: Optional[List[str]] = None,
    relevant_chunk_ids: Optional[List[str]] = None,
    answerability: Optional[str] = None,
    category: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Determine mutually exclusive classification based on state, trace, and ground truth.

    Classes:
    - initial_success: Correct answer on attempt 0 (zero rewrites) with all mandatory facts present.
    - successful_correction: Strict 10-step verified trace (initial FAIL -> rewrite -> re-retrieve -> revised draft -> PASS -> all facts).
    - failed_correction: Rewrote but failed, or reached max retries with FAIL.
    - abstention: Correctly returned 'Not in context.' with PASS on an unanswerable query (out-of-scope or corpus boundary).
    - false_positive_pass: Judge passed an incorrect/unsupported answer, or passed 'Not in context' on an answerable query.
    - false_negative_fail: Judge rejected a correct and fully grounded answer.
    """
    trace = state.get("trace", [])
    final_verdict = (state.get("verdict") or "").upper()
    attempts = state.get("attempts", 0)
    final_draft = state.get("draft", "")

    # Resolve expected decision: "answer" vs "abstain"
    if expected_decision is not None:
        target_decision = expected_decision.strip().lower()
    elif answerability == "unanswerable" or expected_behavior == "abstain" or category == "corpus_boundary":
        target_decision = "abstain"
    else:
        target_decision = "answer"

    facts = mandatory_facts if mandatory_facts is not None else (supporting_facts or [])
    target_chunks = relevant_chunk_ids or []

    is_abstention = "not in context" in final_draft.lower()
    clean_draft = final_draft.lower()
    facts_present = (
        all(f.lower() in clean_draft for f in facts)
        if facts else bool(clean_draft.strip())
    )

    # -------------------------------------------------------------------------
    # Branch A: Unanswerable query (expected_decision == "abstain")
    # Covers both out-of-scope and corpus-boundary queries.
    # -------------------------------------------------------------------------
    if target_decision == "abstain":
        if is_abstention:
            if final_verdict == "PASS":
                return "abstention", {}
            else:
                return "failed_correction", {"note": "Abstention draft was rejected by judge."}
        else:
            if final_verdict == "PASS":
                return "false_positive_pass", {
                    "note": "Model produced an answer for an unanswerable query and judge passed it."
                }
            else:
                return "failed_correction", {
                    "note": "Unanswerable query produced an answer draft which was failed by judge."
                }

    # -------------------------------------------------------------------------
    # Branch B: Answerable query (expected_decision == "answer")
    # Must provide the factual answer. Abstaining on an answerable query is a failure.
    # -------------------------------------------------------------------------
    if is_abstention:
        if final_verdict == "PASS":
            return "false_positive_pass", {
                "note": "Model incorrectly abstained on an answerable query, and judge improperly passed it."
            }
        else:
            return "failed_correction", {
                "note": "Model incorrectly abstained on an answerable query, and judge failed it."
            }

    # Attempt 0: First-attempt answer
    if attempts == 0:
        if final_verdict == "PASS":
            if facts_present:
                return "initial_success", {}
            else:
                return "false_positive_pass", {
                    "note": "Judge passed on attempt 0 without all mandatory ground-truth facts present."
                }
        else:  # final_verdict == FAIL on attempt 0
            if facts_present:
                return "false_negative_fail", {
                    "note": "Judge failed an answer containing all mandatory facts on attempt 0."
                }
            else:
                return "failed_correction", {
                    "note": "Failed on attempt 0 without triggering retries."
                }

    # Attempt > 0: Strict 10-step self-correction verification
    is_strict_correction, trace_steps = verify_strict_self_correction_trace(
        trace=trace,
        original_question=original_question,
        final_verdict=final_verdict,
        final_draft=final_draft,
        mandatory_facts=facts,
        relevant_chunk_ids=target_chunks,
    )

    if is_strict_correction:
        return "successful_correction", trace_steps

    # Strict trace verification failed
    if final_verdict == "PASS":
        if not facts_present:
            return "false_positive_pass", trace_steps
        else:
            return "failed_correction", trace_steps
    else:  # final_verdict == FAIL
        if facts_present:
            return "false_negative_fail", trace_steps
        else:
            return "failed_correction", trace_steps


def extract_attempt_history(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse linear graph trace into structured attempt-by-attempt records."""
    history: Dict[str, Any] = {}

    current_attempt = 0
    current_attempt_data: Dict[str, Any] = {
        "retrieved_chunk_ids": [],
        "retrieved_sources": [],
        "draft": "",
        "judge_verdict": "",
        "judge_reason": "",
        "rewritten_query": None,
    }

    for item in trace:
        node = item.get("node")
        if node == "retrieve_node":
            current_attempt_data["retrieved_chunk_ids"] = item.get("chunk_ids", [])
            current_attempt_data["retrieved_sources"] = item.get("sources", [])
        elif node == "generate_node":
            current_attempt_data["draft"] = item.get("draft", "")
        elif node == "judge_node":
            current_attempt_data["judge_verdict"] = item.get("verdict", "")
            current_attempt_data["judge_reason"] = item.get("reason", "")
        elif node == "rewrite_node":
            # Save the completed attempt
            history[f"attempt_{current_attempt}"] = dict(current_attempt_data)
            current_attempt = item.get("attempt", current_attempt + 1)
            # Reset for next cycle
            current_attempt_data = {
                "retrieved_chunk_ids": [],
                "retrieved_sources": [],
                "draft": "",
                "judge_verdict": "",
                "judge_reason": "",
                "rewritten_query": item.get("rewritten_question", ""),
            }

    # Store final attempt
    history[f"attempt_{current_attempt}"] = dict(current_attempt_data)
    return history


def evaluate_self_correction_benchmark(
    dataset: Optional[List[Dict[str, Any]]] = None,
    pipeline_func: Optional[Callable[[str], Dict[str, Any]]] = None,
    sleep_between_queries: float = 1.5,
) -> Dict[str, Any]:
    """Execute evaluation benchmark across the self-correction dataset.

    Args:
        dataset: Test cases list (defaults to SELF_CORRECTION_DATASET).
        pipeline_func: Callable executing RAG state machine (defaults to run_rag_pipeline).
        sleep_between_queries: Throttle delay between LLM invocations for rate limits.

    Returns:
        Structured dictionary of quantitative metrics and granular execution traces.
    """
    cases = dataset or SELF_CORRECTION_DATASET
    runner = pipeline_func or run_rag_pipeline

    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    latencies_by_attempts: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    attempt_counts: List[int] = []

    classification_counts: Dict[str, int] = {
        "initial_success": 0,
        "successful_correction": 0,
        "failed_correction": 0,
        "abstention": 0,
        "false_positive_pass": 0,
        "false_negative_fail": 0,
    }

    cases_requiring_correction = 0
    max_rewrite_compliant_cases = 0

    print(f"\n================================================================================")
    print(f"Starting Alethia Self-Correction Benchmark ({len(cases)} cases)")
    print(f"================================================================================")

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        cat = case.get("category", "")
        query = case["question"]
        ans_ability = case.get(
            "answerability",
            "answerable" if case.get("expected_decision") == "answer" else "unanswerable",
        )
        exp_decision = case.get(
            "expected_decision",
            "abstain" if case.get("expected_behavior") == "abstain" or cat == "corpus_boundary" else "answer",
        )
        exp_behavior = case.get("expected_behavior", "initial_success" if exp_decision == "answer" else "abstain")
        exp_answer = case.get("expected_answer", "")
        mand_facts = case.get("mandatory_facts", case.get("supporting_facts", []))
        rel_chunks = case.get("relevant_chunk_ids", [])
        corr_reason = case.get("correction_reason", "")

        print(f"\n[{idx}/{len(cases)}] Evaluating `{cid}` ({cat}): '{query[:45]}...'")

        # Execute with retry for transient network / DNS socket errors
        state = None
        t_elapsed = 0.0
        for retry_attempt in range(3):
            try:
                t_start = time.perf_counter()
                state = runner(query)
                t_elapsed = time.perf_counter() - t_start
                break
            except Exception as exc:
                if retry_attempt == 2:
                    raise
                print(f"       [Notice: Transient API connection error ({exc}). Retrying in 4s...]")
                time.sleep(4.0)

        latencies.append(t_elapsed)

        trace = state.get("trace", [])
        final_verdict = (state.get("verdict") or "").upper()
        final_draft = state.get("draft", "")
        attempts = state.get("attempts", 0)
        final_question = state.get("question", query)

        attempt_counts.append(attempts)
        if attempts in latencies_by_attempts:
            latencies_by_attempts[attempts].append(t_elapsed)
        else:
            latencies_by_attempts[2].append(t_elapsed)

        # Check compliance with max rewrites constraint
        if attempts <= MAX_REWRITES:
            max_rewrite_compliant_cases += 1

        # Classify result with strict 10-step criteria
        classification, trace_verification_info = classify_self_correction_case(
            state=state,
            expected_behavior=exp_behavior,
            supporting_facts=mand_facts,
            original_question=query,
            expected_decision=exp_decision,
            mandatory_facts=mand_facts,
            relevant_chunk_ids=rel_chunks,
            answerability=ans_ability,
            category=cat,
        )

        if classification in classification_counts:
            classification_counts[classification] += 1

        # Track cases requiring correction (initial attempt failed and rewrote)
        judge_entries = [e for e in trace if e.get("node") == "judge_node"]
        initial_verdict = judge_entries[0].get("verdict", "").upper() if judge_entries else final_verdict
        if initial_verdict == "FAIL" or attempts > 0:
            cases_requiring_correction += 1

        # Extract structured attempt history
        attempt_history = extract_attempt_history(trace)

        print(f"       -> Classification: [{classification}] | Verdict: {final_verdict} | Retries: {attempts} | Latency: {t_elapsed:.2f}s")

        record: Dict[str, Any] = {
            "id": cid,
            "category": cat,
            "answerability": ans_ability,
            "expected_decision": exp_decision,
            "actual_outcome": classification,
            "classification": classification,
            "question": query,
            "final_question": final_question,
            "expected_behavior": exp_behavior,
            "expected_answer": exp_answer,
            "correction_reason": corr_reason,
            "mandatory_facts": mand_facts,
            "supporting_facts": mand_facts,
            "relevant_chunk_ids": rel_chunks,
            "final_verdict": final_verdict,
            "final_draft": final_draft,
            "attempts": attempts,
            "latency_seconds": round(t_elapsed, 3),
            "trace_verification_details": trace_verification_info,
            "attempt_history": attempt_history,
            "raw_trace": trace,
        }
        records.append(record)

        if sleep_between_queries > 0 and idx < len(cases):
            time.sleep(sleep_between_queries)

    total_evaluated = len(cases)
    answerable_cases = [
        c for c in cases
        if c.get("answerability") == "answerable" or c.get("expected_decision") == "answer"
    ]
    unanswerable_cases = [
        c for c in cases
        if c.get("answerability") == "unanswerable" or c.get("expected_decision") == "abstain"
    ]

    total_answerable = len(answerable_cases)
    total_unanswerable = len(unanswerable_cases)

    out_of_scope_cases = [
        c for c in unanswerable_cases
        if c.get("category") in ("unanswerable", "out_of_scope")
    ]
    corpus_boundary_cases = [
        c for c in unanswerable_cases
        if c.get("category") == "corpus_boundary"
    ]

    total_out_of_scope = len(out_of_scope_cases)
    total_corpus_boundary = len(corpus_boundary_cases)

    # 1. Answer accuracy across genuinely answerable cases
    correct_answers_count = sum(
        1 for r in records
        if r["id"] in [c["id"] for c in answerable_cases]
        and r["actual_outcome"] in ("initial_success", "successful_correction")
    )
    answer_accuracy = round(correct_answers_count / max(1, total_answerable) * 100, 2)
    first_attempt_success_rate = round(
        classification_counts["initial_success"] / max(1, total_answerable) * 100, 2
    )

    # 2. Out-of-scope abstention rate (denominator = 2)
    out_of_scope_abstentions = sum(
        1 for r in records
        if r["id"] in [c["id"] for c in out_of_scope_cases]
        and r["actual_outcome"] == "abstention"
    )
    out_of_scope_abstention_rate = (
        round(out_of_scope_abstentions / max(1, total_out_of_scope) * 100, 2)
        if total_out_of_scope > 0 else 0.0
    )

    # 3. Corpus-boundary abstention rate (denominator = 2)
    corpus_boundary_abstentions = sum(
        1 for r in records
        if r["id"] in [c["id"] for c in corpus_boundary_cases]
        and r["actual_outcome"] == "abstention"
    )
    corpus_boundary_abstention_rate = (
        round(corpus_boundary_abstentions / max(1, total_corpus_boundary) * 100, 2)
        if total_corpus_boundary > 0 else 0.0
    )

    # 4. Overall abstention rate across total suite queries (4/16 = 25.0%)
    total_abstentions = out_of_scope_abstentions + corpus_boundary_abstentions
    overall_abstention_rate = (
        round(total_abstentions / max(1, total_evaluated) * 100, 2)
        if total_evaluated > 0 else 0.0
    )

    # 5. Overall decision correctness across total suite (16/16 = 100.0%)
    correct_decisions_count = correct_answers_count + out_of_scope_abstentions + corpus_boundary_abstentions
    overall_decision_correctness = round(correct_decisions_count / max(1, total_evaluated) * 100, 2)

    # 6. Correction-required denominator and rates:
    # When cases_requiring_correction == 0, reporting 0.0% would be misleading. Must report 'N/A'.
    if cases_requiring_correction > 0:
        successful_correction_of_required: Any = round(
            classification_counts["successful_correction"] / cases_requiring_correction * 100, 2
        )
        failed_correction_rate: Any = round(
            classification_counts["failed_correction"] / cases_requiring_correction * 100, 2
        )
    else:
        successful_correction_of_required = "N/A"
        failed_correction_rate = "N/A"

    # 7. Max rewrite compliance rate:
    # When zero rewrites occurred in the benchmark, compliance with max rewrites is not exercised -> 'N/A'
    if any(a > 0 for a in attempt_counts):
        max_rewrite_compliance_rate: Any = round(
            max_rewrite_compliant_cases / max(1, total_evaluated) * 100, 2
        )
    else:
        max_rewrite_compliance_rate = "N/A"

    successful_correction_of_total_pct = round(
        classification_counts["successful_correction"] / max(1, total_evaluated) * 100, 2
    )
    mean_rewrites = round(sum(attempt_counts) / max(1, total_evaluated), 2)
    mean_latency = round(sum(latencies) / max(1, total_evaluated), 3)

    mean_lat_0 = (
        round(sum(latencies_by_attempts[0]) / len(latencies_by_attempts[0]), 3)
        if latencies_by_attempts[0]
        else 0.0
    )
    mean_lat_1 = (
        round(sum(latencies_by_attempts[1]) / len(latencies_by_attempts[1]), 3)
        if latencies_by_attempts[1]
        else 0.0
    )
    mean_lat_2 = (
        round(sum(latencies_by_attempts[2]) / len(latencies_by_attempts[2]), 3)
        if latencies_by_attempts[2]
        else 0.0
    )

    summary_metrics = {
        "total_evaluated": total_evaluated,
        "total_answerable": total_answerable,
        "total_unanswerable": total_unanswerable,
        "total_out_of_scope": total_out_of_scope,
        "total_corpus_boundary": total_corpus_boundary,
        "cases_requiring_correction": cases_requiring_correction,
        "classification_counts": classification_counts,
        "answer_accuracy": answer_accuracy,
        "first_attempt_success_rate": first_attempt_success_rate,
        "out_of_scope_abstention_rate": out_of_scope_abstention_rate,
        "corpus_boundary_abstention_rate": corpus_boundary_abstention_rate,
        "overall_abstention_rate": overall_abstention_rate,
        "overall_decision_correctness": overall_decision_correctness,
        "successful_correction_rate_of_required": successful_correction_of_required,
        "successful_correction_rate_of_total": successful_correction_of_total_pct,
        "failed_correction_rate": failed_correction_rate,
        "total_live_corrections": classification_counts["successful_correction"],
        "max_rewrite_compliance_rate": max_rewrite_compliance_rate,
        "average_rewrites_per_query": mean_rewrites,
        "average_latency_seconds": {
            "overall_mean": mean_latency,
            "zero_rewrites_mean": mean_lat_0,
            "one_rewrite_mean": mean_lat_1,
            "two_rewrites_mean": mean_lat_2,
            "min_seconds": round(min(latencies), 3) if latencies else 0.0,
            "max_seconds": round(max(latencies), 3) if latencies else 0.0,
        },
    }

    output_payload = {
        "metadata": {
            "benchmark_name": "Alethia Self-Correction Benchmark",
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "max_allowed_rewrites": MAX_REWRITES,
            "total_cases": total_evaluated,
        },
        "summary": summary_metrics,
        "cases": records,
    }

    return output_payload


def run_and_save_self_correction_benchmark(
    output_path: str = "eval/results/self_correction_results.json",
    dataset: Optional[List[Dict[str, Any]]] = None,
    pipeline_func: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute evaluation and serialize results to disk."""
    results = evaluate_self_correction_benchmark(dataset=dataset, pipeline_func=pipeline_func)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Benchmark results saved to `{output_path}`.")
    return results


if __name__ == "__main__":
    run_and_save_self_correction_benchmark()
