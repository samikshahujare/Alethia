# ALETHIA — PHASE 4 IMPLEMENTATION & EVIDENCE AUDIT REPORT

## Executive Summary

Phase 4 of Alethia implemented targeted query rewriting, anti-looping safeguards, and evidence-aware LangGraph orchestration. This report presents the technical implementation and the results of a strict empirical audit of live self-correction behavior conducted with real Groq API inference (`openai/gpt-oss-20b`) and the persistent ChromaDB collection (35 chunks across 7 documents).

**Core Audit Finding**:  
While the LangGraph state machine, query rewriting, and anti-looping safeguards function with 100% mechanical fidelity across 216 passing automated tests, **no genuine evidence-driven self-correction was demonstrated in the live experiments**. The experiments produced either:
1. Retrieval of irrelevant new chunks with answers supported only by initial context (`unsuccessful_evidence_driven_correction`), or
2. Reformulation leading to calibrated abstention under Rule 2 (`calibrated_abstention_not_evidence_correction`).

---

## 1. Files Inspected and Modified

| File | Status | Description |
|---|---|---|
| [`src/state.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/src/state.py) | Modified | Extended `GraphState` TypedDict and `create_initial_state()` with `query_history`, `current_rewrite`, `initial_chunk_ids`, `subsequent_chunk_ids`, `newly_retrieved_chunk_ids`, `newly_retrieved_relevant_chunks`, and `total_latency_ms`. |
| [`src/nodes.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/src/nodes.py) | Modified | Implemented `identify_new_chunks()`, `generate_targeted_keyword_fallback()`, normalized textual duplicate detection in `rewrite_node()`, and evidence-aware retrieval tracking in `retrieve_node()`. |
| [`src/prompts.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/src/prompts.py) | Modified | Added `TARGETED_REWRITE_SYSTEM_PROMPT` and `format_targeted_rewrite_prompt()`. Preserved all 5 hardened Judge rules without modification. |
| [`src/graph.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/src/graph.py) | Modified | Updated `run_rag_pipeline()` to track end-to-end execution latency (`total_latency_ms`) and record comprehensive chunk tracking metadata in the completion trace event. |
| [`tests/test_phase4_orchestration.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/tests/test_phase4_orchestration.py) | Modified | Updated tests to reflect normalized textual duplicate detection (case-folding and punctuation stripping) rather than semantic embeddings. |
| [`eval/results/phase4_live_correction.json`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/results/phase4_live_correction.json) | Created | Dedicated JSON artifact storing audited execution counts, chunk IDs, source excerpts, and the 7-condition checklist. |

---

## 2. Actual Live Execution Counts

All counts reflect actual recorded events from the execution traces:

| Metric | Case 1 (`LIVE-P4-01`: Adam Second-Moment) | Case 2 (`LIVE-P4-02`: Olympus Mons) | Case 3 (`LIVE-P4-03`: Attention Scaling) |
|---|:---:|:---:|:---:|
| **Query** | *"How does Adam calculate its second moment vector and why does it fail with standard L2 weight decay?"* | *"Why is Olympus Mons classified as a shield volcano, and what is its height in kilometers?"* | *"What is the sequence alignment division factor in self-attention, and what does it prevent?"* |
| **Retrieve-Node Executions** | 3 | 2 | 2 |
| **Generate-Node Executions** | 3 | 2 | 2 |
| **Judge-Node Executions** | 3 | 2 | 2 |
| **Rewrite-Node Executions** | 2 | 1 | 1 |
| **Total Attempts (Rewrites)** | 2 | 1 | 1 |
| **Final Verdict** | **PASS** | **PASS** (Rule 2 Abstain) | **PASS** (Rule 2 Abstain) |

---

## 3. Detailed Audit of New Relevant Evidence (Adam Second-Moment Query)

### Chunk Retrieval Breakdown
- **Initial Retrieved Chunk IDs (Attempt 0)**:
  1. `optimizers_guide_txt_chunk_0002`
  2. `optimizers_guide_txt_chunk_0003`
  3. `optimizers_guide_txt_chunk_0001`
  4. `machine_learning_fundamentals_txt_chunk_0004`
- **Post-Rewrite Retrieved Chunk IDs (Attempt 2)**:
  1. `optimizers_guide_txt_chunk_0002`
  2. `optimizers_guide_txt_chunk_0001`
  3. `optimizers_guide_txt_chunk_0003`
  4. `optimizers_guide_txt_chunk_0000`
- **Genuinely New Chunk ID**:
  `optimizers_guide_txt_chunk_0000`

### Inspection of `optimizers_guide_txt_chunk_0000`
```text
Optimization algorithms adjust neural network parameters to minimize empirical loss functions over training steps. The baseline algorithm is Stochastic Gradient Descent (SGD), which updates weights along the steepest descent vector of mini-batch gradients. Incorporating classical momentum or Nesterov accelerated momentum dampens pathological oscillations along steep ravines.
```
- **Does it support the missing second-moment formula?**  
  **NO.** The chunk discusses Stochastic Gradient Descent (SGD), steepest descent, and classical/Nesterov momentum for dampening ravine oscillations. It contains zero formulas or explanations regarding Adam's second-moment vector or L2 weight decay.

### Source Excerpts Actually Supporting the Final Answer
The final answer was supported entirely by chunks retrieved on **Attempt 0**:
1. From `optimizers_guide_txt_chunk_0002` (Retrieved Attempt 0):
   > *"Standard Adam uses default hyperparameter configurations: beta_1 equals 0.9 for first-moment momentum, beta_2 equals 0.999 for second-moment variance tracking, and epsilon equals 1e-8 to prevent division by zero. However, standard Adam implements weight decay incorrectly when paired with L2 regularization."*
2. From `optimizers_guide_txt_chunk_0003` (Retrieved Attempt 0):
   > *"The AdamW optimizer resolves this flaw by decoupling weight decay from gradient updates, directly decaying weights proportionally at each step."*

Because the genuinely new chunk was irrelevant and the final answer relied solely on Attempt 0 chunks, `LIVE-P4-01` did **not** demonstrate evidence-driven self-correction.

---

## 4. Strict Self-Correction Condition Checklist

A genuine self-correction must satisfy all seven conditions without exception:

| # | Condition | Case 1 (`LIVE-P4-01`) | Case 2 (`LIVE-P4-02`) | Case 3 (`LIVE-P4-03`) |
|:---:|---|:---:|:---:|:---:|
| 1 | Initial answer contains unsupported claim or fails to answer required fact | PASS | PASS | PASS |
| 2 | Initial Judge returns `FAIL` | PASS | PASS | PASS |
| 3 | Rewrite node executes | PASS (2 rewrites) | PASS (1 rewrite) | PASS (1 rewrite) |
| 4 | Retrieval executes after rewriting | PASS | PASS | PASS |
| 5 | **At least one genuinely new relevant in-corpus chunk retrieved** | **FAIL** (`chunk_0000` irrelevant) | **FAIL** (`ml_chunk_0002` irrelevant) | **FAIL** (0 new chunks) |
| 6 | **Final answer uses that new evidence** | **FAIL** (used Attempt 0 chunks) | **FAIL** (abstained) | **FAIL** (abstained) |
| 7 | Final Judge returns `PASS` | PASS | PASS (Rule 2) | PASS (Rule 2) |
| **All** | **Strict Verification Status** | **FAIL** | **FAIL** | **FAIL** |

---

## 5. Duplicate Prevention: Implementation vs Claims

The duplicate prevention mechanism in `rewrite_node()` was audited against its code:

```python
norm_history = {re.sub(r"[^\w\s]", "", q.lower()).strip() for q in query_history if q}
clean_rewritten = raw_rewritten.strip().strip('"').strip("'")
norm_candidate = re.sub(r"[^\w\s]", "", clean_rewritten.lower()).strip()

if not clean_rewritten or norm_candidate in norm_history or norm_candidate == "":
    clean_rewritten = generate_targeted_keyword_fallback(...)
```

### Supported Capabilities
1. **Exact Textual Duplicate Detection**: Identifies exact string repetitions across `query_history`.
2. **Normalized Textual Duplicate Detection**: Normalizes strings by lowercasing, stripping leading/trailing punctuation and whitespace, and removing non-alphanumeric symbols (`re.sub(r"[^\w\s]", "", q.lower())`).

### Non-Supported Capabilities
- **Semantic Similarity Detection**: The engine does **not** compute embedding cosine similarity or semantic entailment between queries. Describing string normalization as "semantic duplicate detection" is technically inaccurate and has been corrected in the test suite and documentation.

---

## 6. Final Classification

$$\mathbf{NO\ GENUINE\ EVIDENCE-DRIVEN\ CORRECTION\ DEMONSTRATED}$$

### Empirical Reasons
1. **Hybrid Retrieval Precision**: In the 35-chunk corpus, Okapi BM25 + dense search + RRF retrieves the target chunks on Attempt 0 for answerable questions.
2. **Hardened Judge Rule 2 Compliance**: When context is absent, the generator correctly outputs `"Not in context."`, and the Judge appropriately awards `PASS` under Rule 2 without triggering retries.
3. **Absence of Staged Evidence Discovery**: In all multi-attempt runs, newly retrieved chunks were either nonexistent (`LIVE-P4-03`) or irrelevant (`LIVE-P4-01`, `LIVE-P4-02`). The pipeline demonstrated mechanical loop execution and calibrated abstention, but not evidence-driven correction.

---

## 7. ChromaDB and Test Verification

### Test Suite Execution
```
pytest tests/ -q
........................................................................ [ 33%]
........................................................................ [ 66%]
........................................................................ [100%]
216 passed in 111.77s (0:01:51)
Exit Code: 0
```
- Total tests: **216 passed** across 12 test files.
- 0 failed, 0 skipped, 0 warnings.

### ChromaDB Integrity
- Direct verification via `store.collection.count()`: **Exactly 35 chunks**.
- All operations maintain strict read-only access. No document mutations occurred.
- All historical evaluation artifacts (`eval_benchmark_results.json`, `self_correction_results.json`, `step4_live_correction.json`, `step4_2_hardened_live_correction.json`) remain intact.

