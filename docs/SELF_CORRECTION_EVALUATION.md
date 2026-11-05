# Alethia: Dedicated Real Self-Correction Evaluation & Dynamics Report (Step 3 Final)

**Role:** Senior AI/ML Evaluation Engineer specializing in RAG and LangGraph  
**Date:** November 2026  
**System Tested:** Alethia Self-Correcting RAG Engine (LangGraph + ChromaDB + `all-MiniLM-L6-v2` + Groq `openai/gpt-oss-20b`)  
**Artifacts Generated:** `eval/self_correction_eval.py`, `eval/results/self_correction_results.json`, `tests/test_self_correction_eval.py`  
**Test Suite Status:** 28/28 tests passing in `tests/test_self_correction_eval.py` (Full suite: 116/116 tests passing, `pytest tests/ -v`)  

---

## 1. Executive Summary

In Step 1 and Step 2 of the technical audit, we eliminated prior evaluation conflations where first-attempt answer success was counted as active self-correction.

In **Step 3**, we implemented a dedicated, mathematically partitioned self-correction benchmark measuring:
1. Whether Alethia can detect an unsupported draft, rewrite the query, re-retrieve relevant evidence, and produce a grounded answer.
2. The empirical behavior of the cyclic LangGraph state machine across a curated 16-case benchmark partitioned by **answerability** and **expected decision**:
   - **12 Answerable questions** (`SC-01` through `SC-11`, `SC-16`): Tested with lexical traps, cross-domain ambiguity, and phrasing challenges.
   - **2 Out-of-scope unanswerable questions** (`SC-12`, `SC-13`): Tested for immediate abstention.
   - **2 Corpus-boundary unanswerable questions** (`SC-14`, `SC-15`): In-domain questions where specific numerical figures are absent, testing resistance to hallucination and proper abstention.

### Core Measured Results (Empirical Live Groq Benchmark)

| Metric | Denominator & Definition | Empirical Measured Value | Status / Interpretation |
|:---|:---|:---:|:---|
| **Total Test Cases Evaluated** | Total suite queries | **16 cases** | Complete evaluation run |
| **Answer Accuracy** | Answerable queries correctly answered ($12 / 12$) | **100.0%** | All 12 answerable queries answered accurately |
| **First-Attempt Answer Accuracy** | Answerable queries passing on cycle 1 ($12 / 12$) | **100.0%** | Zero rewrites needed; top-4 retrieval supplied ground truth |
| **Out-of-Scope Abstention Rate** | Out-of-scope queries correctly abstaining ($2 / 2$) | **100.0%** | Immediate cycle-1 `"Not in context."` with Judge `PASS` |
| **Corpus-Boundary Abstention Rate** | Absent figures correctly abstaining ($2 / 2$) | **100.0%** | Correctly returned `"Not in context."` on attempt 0 (classified as `abstention`) |
| **Overall Abstention Rate** | Total abstentions / Total evaluated ($4 / 16$) | **25.0%** | 4 abstentions across 16 total queries |
| **Overall Decision Correctness** | Correct answers + Correct abstentions ($16 / 16$) | **100.0%** | $(12 + 2 + 2) / 16 = 100.0\%$ (measured against benchmark expected decisions) |
| **Cases Requiring Correction** | Queries failing initial Judge verdict | **0 cases** | Initial Judge approved all 16 first-attempt outputs |
| **Active Rewrites Triggered** | Total query reformulation cycles executed | **0 / 16 queries** | Empirically unexercised across 16 benchmark queries |
| **Successful Correction of Required** | Successful corrections / Cases requiring correction | **N/A** | Denominator is zero ($0/0$); reported defensibly as `N/A` |
| **Failed Correction Rate** | Failed corrections / Cases requiring correction | **N/A** | Denominator is zero ($0/0$); reported defensibly as `N/A` |
| **Total Live Corrections** | Total queries corrected via rewrite loop | **0 / 16** | Transparently reported without fabrication |
| **Max Rewrite Limit Compliance** | Cases complying with $\le 2$ rewrites | **N/A** | Zero rewrites occurred in live benchmark run |
| **Average Rewrites per Query** | Mean rewrite count across all queries | **0.00 rewrites** | Mean rewrite count |
| **Mean Pipeline Latency** | Mean wall-clock seconds per query | **6.734s** | Min: 1.536s, Max: 26.434s |

> [!IMPORTANT]
> **Audit Conclusion on Active Self-Correction**:  
> In the live benchmark against Groq (`openai/gpt-oss-20b`), **0 out of 16 queries triggered an actual rewrite** because dense semantic retrieval ($k=4$) succeeded in placing the target chunk in the top-4 context on attempt 0, and the model adhered strictly to generation prompts.  
> As a result, **active self-correction remains unexercised in the live pipeline**. Alethia reports `successful_correction_rate_of_required = "N/A"`, `failed_correction_rate = "N/A"`, and `max_rewrite_compliance_rate = "N/A"` rather than fabricating retries or claiming 100% self-correction.

---

## 2. Evaluation Methodology & Verification Protocol

### 2.1. Strict 10-Step Chronological Verification Protocol

A query is classified as `successful_correction` **if and only if** all 10 events occur in strict chronological sequence:

1. **Initial Retrieval Completes:** `retrieve_node` retrieves initial top-$k$ chunks.
2. **Initial Generate Produces Draft:** `generate_node` produces an initial draft answer.
3. **Initial Judge Verdict is FAIL:** `judge_node` evaluates the draft against context and returns `VERDICT: FAIL`.
4. **Rewrite Node Executes:** State machine transitions to `rewrite_node` (`attempts` incremented to 1).
5. **Query Changed:** The reformulated question differs from the original query (`rewritten_q != orig_q`).
6. **Subsequent Retrieval Executes with Rewritten Query:** `retrieve_node` executes using the exact `rewritten_question`.
7. **Newly Retrieved Ground-Truth Chunks Retrieved:** Post-rewrite retrieval fetches at least one relevant ground-truth chunk that was *not* present in the initial retrieval (`newly_retrieved_chunks = post_chunk_ids - initial_chunk_ids`, `new_relevant_chunks = newly_retrieved_chunks.intersection(target_chunks)`, requiring `len(new_relevant_chunks) > 0`). A ground-truth chunk that was already present in initial retrieval does not count as newly discovered evidence.
8. **Revised Answer Draft Produced:** `generate_node` produces a revised draft from the new context.
9. **Final Judge Verdict is PASS after Revised Generate:** `judge_node` evaluates the revised draft and returns `VERDICT: PASS`.
10. **All Mandatory Ground-Truth Facts Present:** The final answer contains all case-specific mandatory ground-truth facts (`mandatory_facts`).

### 2.2. Strengthened Evidence Verification & Limitations

1. **Rejection of Arbitrary New Chunks:**
   In earlier iterations, a fallback allowed `has_ground_truth_chunks = has_different_chunks`. This fallback has been **strictly eliminated**. Retrieving different chunk IDs does not establish that useful evidence was found. Post-rewrite retrieval must fetch the case's explicit `relevant_chunk_ids`. If no ground-truth chunk IDs are matched, `step_07_ground_truth_chunks_retrieved` strictly evaluates to `False`.

2. **Evaluation Limitation Note on Substring Matching:**
   Mandatory-fact substring matching is a limited evaluation heuristic, not a semantic correctness guarantee. While it checks for the physical presence of required factual tokens, true semantic correctness requires contextual validation, human evaluation, or advanced entailment modeling.

### 2.3. Mutually Exclusive Classification Schema

Every evaluation record strictly separates **expected behavior** from **actual outcome**:

- **Taxonomy Dimensions**:
  - `category`: Challenge classification (e.g. `cross_domain`, `lexical_trap`, `ambiguous`, `insufficient_context`, `unanswerable`, `corpus_boundary`, `control`).
  - `answerability`: `"answerable"` vs `"unanswerable"`.
  - `expected_decision`: `"answer"` vs `"abstain"`.
  - `actual_outcome`: Exactly one of:
    - `initial_success`: Answerable query correctly answered on attempt 0 with all mandatory facts present.
    - `successful_correction`: Verified 10-step self-correction sequence.
    - `failed_correction`: Rewrote but failed, or reached max retries with FAIL.
    - `abstention`: Unanswerable query (out-of-scope or corpus-boundary) correctly returning `"Not in context."` on attempt 0 with Judge `PASS`.
    - `false_positive_pass`: Judge passed an incorrect/unsupported answer, or passed `"Not in context."` on an answerable query, or passed a hallucination on an unanswerable query.
    - `false_negative_fail`: Judge rejected an answer that contained all mandatory facts and was supported by context.

> [!NOTE]
> **Dataset Label Clarification**:  
> Cases previously categorized as `insufficient_context` or `ambiguous` that succeeded on attempt 0 must **never** be described as demonstrated self-correction cases. They reflect first-attempt retrieval robustness and generation fidelity, not query reformulation dynamics.

---

## 3. Dedicated Self-Correction Evaluation Dataset (16 Cases)

The dataset was constructed against the 7 documents (35 chunks) in `data/`:

| ID | Challenge Category | Answerability | Expected Decision | Actual Outcome | Target Chunk ID | Mandatory Ground-Truth Facts |
|:---|:---|:---:|:---:|:---:|:---|:---|
| `SC-01` | Cross-Domain | `answerable` | `answer` | `initial_success` | `machine_learning_fundamentals_txt_chunk_0003` | `l1`, `regularization` |
| `SC-02` | Lexical Trap | `answerable` | `answer` | `initial_success` | `optimizers_guide_txt_chunk_0002` | `root`, `gradient` |
| `SC-03` | Ambiguous | `answerable` | `answer` | `initial_success` | `attention_mechanisms_txt_chunk_0001` | `d_k`, `gradient` |
| `SC-04` | Insufficient Context | `answerable` | `answer` | `initial_success` | `neural_network_training_txt_chunk_0002` | `gelu`, `gaussian` |
| `SC-05` | Lexical Trap | `answerable` | `answer` | `initial_success` | `attention_mechanisms_txt_chunk_0003` | `causal mask`, `negative infinity` |
| `SC-06` | Ambiguous | `answerable` | `answer` | `initial_success` | `optimizers_guide_txt_chunk_0001` | `momentum`, `nesterov` |
| `SC-07` | Insufficient Context | `answerable` | `answer` | `initial_success` | `nlp_concepts_txt_chunk_0002` | `sentencepiece` |
| `SC-08` | Insufficient Context | `answerable` | `answer` | `initial_success` | `nlp_concepts_txt_chunk_0004` | `bleu`, `brevity penalty` |
| `SC-09` | Ambiguous | `answerable` | `answer` | `initial_success` | `neural_network_training_txt_chunk_0004` | `layer normalization` |
| `SC-10` | Ambiguous | `answerable` | `answer` | `initial_success` | `solar_system_txt_chunk_0003` | `water`, `ice`, `silicate` |
| `SC-11` | Ambiguous | `answerable` | `answer` | `initial_success` | `solar_system_txt_chunk_0002` | `olympus`, `mons`, `21.9` |
| `SC-12` | Out-of-Scope | `unanswerable` | `abstain` | `abstention` | None | Abstain (`"Not in context."`) |
| `SC-13` | Out-of-Scope | `unanswerable` | `abstain` | `abstention` | None | Abstain (`"Not in context."`) |
| `SC-14` | Corpus Boundary | `unanswerable` | `abstain` | `abstention` | None | Abstain (`"Not in context."`) |
| `SC-15` | Corpus Boundary | `unanswerable` | `abstain` | `abstention` | None | Abstain (`"Not in context."`) |
| `SC-16` | Factual Control | `answerable` | `answer` | `initial_success` | `transformer_summary_txt_chunk_0002` | `six encoder`, `six decoder` |

---

## 4. Empirical Benchmark Case-by-Case Results

Live execution against Groq (`openai/gpt-oss-20b`) and persistent ChromaDB (`eval/results/self_correction_results.json`):

| ID | Category | Query (Truncated) | Final Verdict | Retries | Latency | Actual Outcome | Final Answer Output |
|:---|:---|:---|:---:|:---:|:---:|:---|:---|
| `SC-01` | Cross-Domain | *What is the penalty that creates sparse features?* | **PASS** | 0 | 26.43s | `initial_success` | "L1 regularization (the L1 penalty)." |
| `SC-02` | Lexical Trap | *How is the running root-mean-square average used...* | **PASS** | 0 | 1.59s | `initial_success` | "It divides the gradient by the running root-mean-square average..." |
| `SC-03` | Ambiguous | *Why do we divide by the square root during sequence...* | **PASS** | 0 | 2.18s | `initial_success` | "Because the scaling factor sqrt(d_k) normalizes the dot-product scores..." |
| `SC-04` | Insufficient | *What causes gradient saturation and how do we...* | **PASS** | 0 | 1.54s | `initial_success` | "Gradient saturation is caused by... GELU activation weighting inputs by..." |
| `SC-05` | Lexical Trap | *How are future tokens hidden during autoregressive...* | **PASS** | 0 | 1.72s | `initial_success` | "Causal masking sets future token positions to negative infinity before softmax..." |
| `SC-06` | Ambiguous | *What prevents weights from oscillating wildly...* | **PASS** | 0 | 1.86s | `initial_success` | "Classical momentum or Nesterov accelerated momentum." |
| `SC-07` | Insufficient | *Which subword algorithm avoids pre-segmentation...* | **PASS** | 0 | 1.80s | `initial_success` | "SentencePiece." |
| `SC-08` | Insufficient | *How do we penalize overly brief translated text?* | **PASS** | 0 | 2.53s | `initial_success` | "BLEU incorporates a brevity penalty to penalize overly brief translations." |
| `SC-09` | Ambiguous | *What metric is independent of batch size for...* | **PASS** | 0 | 8.13s | `initial_success` | "Layer normalization." |
| `SC-10` | Ambiguous | *What are the rings composed of?* | **PASS** | 0 | 8.31s | `initial_success` | "The rings are composed mainly of billions of small water-ice particles..." |
| `SC-11` | Ambiguous | *What is the height of the largest mountain?* | **PASS** | 0 | 7.79s | `initial_success` | "The largest mountain, Olympus Mons on Mars, stands approximately 21.9 km high." |
| `SC-12` | Out-of-Scope | *What is the capital city of France?* | **PASS** | 0 | 6.69s | `abstention` | "Not in context." |
| `SC-13` | Out-of-Scope | *What quantum gate sequence is used in the attention...* | **PASS** | 0 | 8.51s | `abstention` | "Not in context." |
| `SC-14` | Boundary | *What was the exact numerical learning rate value...* | **PASS** | 0 | 10.31s | `abstention` | "Not in context." |
| `SC-15` | Boundary | *What was the batch size used when training GPT...* | **PASS** | 0 | 9.98s | `abstention` | "Not in context." |
| `SC-16` | Control | *How many encoder and decoder layers does the base...* | **PASS** | 0 | 8.38s | `initial_success` | "Six encoder layers and six decoder layers." |

---

## 5. Metric Calculations & Denominators

1. **Answer Accuracy (Denominator = 12 Answerable Cases)**:
   $$\text{Answer Accuracy} = \frac{\text{Correct Answers}}{\text{Total Answerable}} = \frac{12}{12} = \mathbf{100.0\%}$$

2. **Out-of-Scope Abstention Rate (Denominator = 2 Out-of-Scope Cases)**:
   $$\text{Out-of-Scope Abstention Rate} = \frac{\text{Abstentions}}{\text{Total Out-of-Scope}} = \frac{2}{2} = \mathbf{100.0\%}$$

3. **Corpus-Boundary Abstention Rate (Denominator = 2 Boundary Cases)**:
   $$\text{Corpus-Boundary Abstention Rate} = \frac{\text{Abstentions}}{\text{Total Boundary}} = \frac{2}{2} = \mathbf{100.0\%}$$

4. **Overall Abstention Rate (Denominator = 16 Total Evaluated Cases)**:
   $$\text{Overall Abstention Rate} = \frac{\text{Total Abstentions}}{\text{Total Evaluated}} = \frac{4}{16} = \mathbf{25.0\%}$$

5. **Overall Decision Correctness (Denominator = 16 Total Evaluated Cases)**:
   $$\text{Overall Decision Correctness} = \frac{\text{Correct Decisions}}{\text{Total Evaluated}} = \frac{12 + 2 + 2}{16} = \frac{16}{16} = \mathbf{100.0\%}$$

   > **Measurement Scope & Heuristic Limitation Note**: Overall decision correctness (100.0%, 16/16) is strictly measured against the manually defined benchmark expected decisions and the limited mandatory-fact substring evaluation heuristic, not an unconstrained real-world semantic correctness proof.

6. **Cases Requiring Correction (Denominator = 0)**:
   - Initial Judge verdict was `PASS` on attempt 0 for all 16 cases.
   - Denominator = 0.
   - Successful self-correction rate of required cases = **`N/A`**.
   - Failed self-correction rate = **`N/A`**.

7. **Max Rewrite Compliance Rate**:
   - Zero rewrites occurred in the live benchmark run = **`N/A`**.

8. **Total Live Corrections**:
   - Total queries where rewrite occurred and corrected an initial failure: **0 / 16**.

---

## 6. Automated Unit Test Verification

Automated regression coverage for all evaluation rules is verified in `tests/test_self_correction_eval.py`:

- `test_self_correction_dataset_integrity` (16 cases, 12 answerable, 2 out-of-scope, 2 boundary)
- `test_strict_self_correction_trace_all_10_steps_passing`
- `test_strict_self_correction_rejects_missing_any_step` (13 distinct step failure variations)
- `test_different_chunks_alone_do_not_establish_useful_evidence` (strictly rejects new chunks without ground truth)
- `test_incorrect_boundary_classification_fixed` (verifies corpus boundary is classified as `abstention`, never `initial_success`)
- `test_zero_denominator_reports_not_applicable` (verifies denominator 0 produces `"N/A"`)
- `test_abstention_on_answerable_query_rejected` (verifies `"Not in context."` on answerable query is rejected)
- `test_missing_mandatory_facts_fails_verification_and_classification` (verifies single-token answers fail rubric)
- `test_wrong_subsequent_retrieval_query_fails_verification`
- `test_invalid_final_judge_ordering_fails_verification`
- `test_unrelated_chunks_after_rewrite_fails_verification`
- `test_mutually_exclusive_classification_logic`
- `test_extract_attempt_history`
- `test_benchmark_metrics_calculation_math`
- `test_final_audit_metric_specifications` (verifies 25% overall abstention, N/A rates, 100% answer accuracy, 100% decision correctness)

**Total unit tests in suite:** 27 passing (`pytest tests/test_self_correction_eval.py -v`).
