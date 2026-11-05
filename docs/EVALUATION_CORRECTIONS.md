# Alethia Evaluation Corrections & Methodology Realignment (Step 1 & Step 2 Final Audit)

**Date**: November 2, 2026  
**Role**: Independent Information Retrieval & AI/ML Evaluation Auditor  
**Reference Documents**: 
- [`docs/INDEPENDENT_AUDIT_REPORT.md`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/docs/INDEPENDENT_AUDIT_REPORT.md)
- [`eval/dataset.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/dataset.py)
- [`eval/retrieval_eval.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/retrieval_eval.py)
- [`eval/system_eval.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/system_eval.py)
- [`eval/results/eval_benchmark_results.json`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/results/eval_benchmark_results.json)  
**Deliverable Document**: `docs/EVALUATION_CORRECTIONS.md`  

---

## 1. Overview & Scope of Independent Audit

Following the completion of Step 1 (Classification Logic & Reporting Realignment) and Step 2 (Information Retrieval Methodology Realignment), this targeted independent audit verifies:
1. **Mathematical Distinction between Hit Rate@K and Recall@K**:
   Eliminating nomenclature conflation between binary hit success and multi-document recall proportion.
2. **Ground-Truth Independence & Verification**:
   Query-by-query proof that all 22 ground-truth chunk IDs physically exist in ChromaDB, contain genuine textual evidence, and were not assigned post-hoc.
3. **Separate Handling of Unanswerable Queries**:
   Exclusion of the 10 out-of-scope abstention queries from the answerable retrieval recall denominator.
4. **Complete Regression Verification**:
   Executing the full 88-test regression suite to completion without premature termination or test omissions.

---

## 2. Hit Rate@K versus Recall@K Mathematical Realignment

### 2.1. Formal Metric Definitions

In Information Retrieval literature, evaluation over multi-chunk ground truth requires strictly separating binary query success from coverage proportion:

$$\text{Hit Rate@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\Big(|\text{Retrieved}_K(q) \cap \text{Relevant}(q)| \ge 1\Big)$$

$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Retrieved}_K(q) \cap \text{Relevant}(q)|}{|\text{Relevant}(q)|}$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_{\text{first}}(q)}$$

### 2.2. Demonstrative Hand-Calculated Example

Consider a query with 4 ground-truth relevant chunks ($|\text{Relevant}| = 4$):
- Top 3 retrieved chunks contain 2 relevant chunks and 1 irrelevant chunk ($|\text{Retrieved}_3 \cap \text{Relevant}| = 2$).
- **Recall@3**: $2 / 4 = \mathbf{50.0\%}$ (half of all available ground-truth evidence was retrieved).
- **Hit Rate@3**: $\mathbf{100.0\%}$ (at least one relevant chunk was found within Top 3).

Mislabelling Hit Rate as Recall overstates retrieval coverage when queries require or span multiple supporting passages.

### 2.3. Impact on Alethia's 20 Answerable Queries

In Alethia's 20 answerable benchmark queries:
- **18 queries** have a single ground-truth chunk ($|\text{Relevant}| = 1$). For these queries, $\text{Recall@K} = \text{Hit Rate@K}$.
- **2 queries** have multiple ground-truth chunks ($|\text{Relevant}| = 2$):
  - `FAC-05` (L1 vs L2 regularization): `chunk_0003` (L1 Lasso) and `chunk_0004` (L2 Ridge).
  - `AMB-04` (Jupiter's moons): `chunk_0001` (95 moons) and `chunk_0002` (Ganymede).

At Top 1 ($K=1$):
- 15 single-chunk queries retrieved their target chunk at Rank 1 ($\text{Recall@1} = 1.0, \text{Hit@1} = 1$).
- 3 single-chunk queries retrieved their target chunk at Rank 2: `FAC-03`, `AMB-03`, `CHL-01` ($\text{Recall@1} = 0.0, \text{Hit@1} = 0$).
- Both multi-chunk queries (`FAC-05` and `AMB-04`) retrieved 1 relevant chunk at Rank 1.
  - Per-query $\text{Hit@1} = 1$
  - Per-query $\text{Recall@1} = 1 / 2 = 0.50$

**Aggregate Benchmark Scores across all 20 answerable queries**:
- **Hit Rate@1**: $(15 + 2) / 20 = \mathbf{85.0\% (17/20)}$
- **Recall@1**: $(15 \times 1.0 + 2 \times 0.5 + 3 \times 0.0) / 20 = 16.0 / 20 = \mathbf{80.0\%}$
- **Hit Rate@3**: $20 / 20 = \mathbf{100.0\%}$
- **Recall@3**: $(18 \times 1.0 + 2 \times 2/2) / 20 = 20.0 / 20 = \mathbf{100.0\%}$
- **Hit Rate@5**: $20 / 20 = \mathbf{100.0\%}$
- **Recall@5**: $20 / 20 = \mathbf{100.0\%}$
- **Mean Reciprocal Rank (MRR)**: $(17 \times 1.0 + 3 \times 0.5) / 20 = 18.5 / 20 = \mathbf{0.9250}$

---

## 3. Query-by-Query Ground-Truth Independence Audit

Every single ground-truth assignment in [`eval/dataset.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/eval/dataset.py) was audited against the persistent ChromaDB collection. All 22 referenced chunk IDs exist, and their evidence was verified:

| Query ID | Category | Query Summary | Relevant Chunk ID(s) | Exists? | Genuine Evidence in Corpus Passage | First Rank | Top-1 Chunk Found | Hit@1 | Rec@1 |
|:---|:---|:---|:---|:---:|:---|:---:|:---|:---:|:---:|
| `FAC-01` | factual | Transformer encoder/decoder layers | `transformer_summary_txt_chunk_0002` | **True** | "base architecture contains Six encoder layers and six decoder layers" | **1** | `transformer_summary_txt_chunk_0002` | 1 | 1.0 |
| `FAC-02` | factual | Transformer BLEU on En-De | `transformer_summary_txt_chunk_0004` | **True** | "achieving 28.4 BLEU for English-to-German translation" | **1** | `transformer_summary_txt_chunk_0004` | 1 | 1.0 |
| `FAC-03` | factual | Scaled dot-product formula | `attention_mechanisms_txt_chunk_0001` | **True** | "Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d_k)) * V" | **2** | `attention_mechanisms_txt_chunk_0000` (Rank 1)<br>`attention_mechanisms_txt_chunk_0001` (Rank 2) | 0 | 0.0 |
| `FAC-04` | factual | Multi-head attention head count | `attention_mechanisms_txt_chunk_0003` | **True** | "eight parallel attention heads are computed simultaneously" | **1** | `attention_mechanisms_txt_chunk_0003` | 1 | 1.0 |
| `FAC-05` | factual | L1 Lasso vs L2 Ridge difference | `machine_learning_fundamentals_txt_chunk_0003`<br>`machine_learning_fundamentals_txt_chunk_0004` | **True**<br>**True** | Chunk 3: "penalty proportional to absolute value of weights"<br>Chunk 4: "L2 Ridge penalizes squared magnitude of weights" | **1** | `machine_learning_fundamentals_txt_chunk_0004` (Rank 1)<br>`machine_learning_fundamentals_txt_chunk_0003` (Rank 2) | 1 | 0.5 |
| `FAC-06` | factual | Training/val/test split ratio | `machine_learning_fundamentals_txt_chunk_0005` | **True** | "partitioned into an 80/10/10 ratio for training, validation, and testing" | **1** | `machine_learning_fundamentals_txt_chunk_0005` | 1 | 1.0 |
| `FAC-07` | factual | GELU vs ReLU Gaussian smoothing | `neural_network_training_txt_chunk_0002` | **True** | "GELU weights inputs by probability under Gaussian cumulative distribution" | **1** | `neural_network_training_txt_chunk_0002` | 1 | 1.0 |
| `FAC-08` | factual | Gradient clipping threshold | `neural_network_training_txt_chunk_0003` | **True** | "maximum norm threshold of 1.0 are scaled proportionally" | **1** | `neural_network_training_txt_chunk_0003` | 1 | 1.0 |
| `FAC-09` | factual | Adam beta_1 and beta_2 defaults | `optimizers_guide_txt_chunk_0002` | **True** | "beta_1 equals 0.9 for first-moment, beta_2 equals 0.999" | **1** | `optimizers_guide_txt_chunk_0002` | 1 | 1.0 |
| `FAC-10` | factual | Jupiter natural moon count | `solar_system_txt_chunk_0001` | **True** | "cataloged 95 officially recognized natural moons orbiting Jupiter" | **1** | `solar_system_txt_chunk_0001` | 1 | 1.0 |
| `AMB-01` | ambiguous | "How many layers?" | `transformer_summary_txt_chunk_0002` | **True** | "base architecture contains Six encoder layers and six decoder layers" | **1** | `transformer_summary_txt_chunk_0002` | 1 | 1.0 |
| `AMB-02` | ambiguous | "What are the beta parameters?" | `optimizers_guide_txt_chunk_0002` | **True** | "beta_1 equals 0.9 for first-moment momentum, beta_2 equals 0.999" | **1** | `optimizers_guide_txt_chunk_0002` | 1 | 1.0 |
| `AMB-03` | ambiguous | "What is the formula?" | `attention_mechanisms_txt_chunk_0001` | **True** | "Softmax(Q * K^T / sqrt(d_k)) * V" | **2** | `attention_mechanisms_txt_chunk_0000` (Rank 1)<br>`attention_mechanisms_txt_chunk_0001` (Rank 2) | 0 | 0.0 |
| `AMB-04` | ambiguous | "Tell me about the moons." | `solar_system_txt_chunk_0001`<br>`solar_system_txt_chunk_0002` | **True**<br>**True** | Chunk 1: "95 officially recognized natural moons orbiting Jupiter"<br>Chunk 2: "Ganymede is the largest moon in the solar system" | **1** | `solar_system_txt_chunk_0001` (Rank 1)<br>`solar_system_txt_chunk_0002` (Rank 2) | 1 | 0.5 |
| `AMB-05` | ambiguous | "What is the split ratio?" | `machine_learning_fundamentals_txt_chunk_0005` | **True** | "80/10/10 ratio for training, validation, and testing" | **1** | `machine_learning_fundamentals_txt_chunk_0005` | 1 | 1.0 |
| `CHL-01` | challenge | AdamW weight decay resolution | `optimizers_guide_txt_chunk_0003` | **True** | "AdamW decouples weight decay from gradient updates, directly decaying weights proportionally" | **2** | `optimizers_guide_txt_chunk_0002` (Rank 1)<br>`optimizers_guide_txt_chunk_0003` (Rank 2) | 0 | 0.0 |
| `CHL-02` | challenge | Perplexity mathematical definition | `nlp_concepts_txt_chunk_0003` | **True** | "Perplexity is mathematically defined as the exponentiated average negative log-likelihood per token" | **1** | `nlp_concepts_txt_chunk_0003` | 1 | 1.0 |
| `CHL-03` | challenge | Transformer BLEU on En-Fr | `transformer_summary_txt_chunk_0004` | **True** | "41.8 BLEU on the WMT 2014 English-to-French benchmark" | **1** | `transformer_summary_txt_chunk_0004` | 1 | 1.0 |
| `CHL-04` | challenge | SentencePiece raw character streaming | `nlp_concepts_txt_chunk_0002` | **True** | "SentencePiece treats input text as a raw stream of characters, including whitespace characters" | **1** | `nlp_concepts_txt_chunk_0002` | 1 | 1.0 |
| `CHL-05` | challenge | Olympus Mons volcano height | `solar_system_txt_chunk_0003` | **True** | "Olympus Mons, a massive shield volcano standing approximately 21.9 kilometers high" | **1** | `solar_system_txt_chunk_0003` | 1 | 1.0 |

### 3.1. Out-of-Scope Queries (`ABS-01` through `ABS-10`)
All 10 unanswerable queries (`ABS-01` to `ABS-10`) have `relevant_source = None`, `relevant_chunk_ids = []`, and `expected_behavior = "abstain"`. They were confirmed excluded from the 20 answerable retrieval queries. Calling `is_chunk_relevant` on any of them returns `(False, "unanswerable_query")`.

---

## 4. Reconciled Metrics: Provisional vs Audited Baseline

| Retrieval Metric | Provisional Claim | Audited & Verified Metric | Methodological Cause |
|:---|:---:|:---:|:---|
| **Hit Rate@1** | 90.0% (18/20) | **85.0% (17/20)** | Corrected generic keyword false-positive on `CHL-01`; true target chunk is Rank 2. |
| **Hit Rate@3** | 100.0% (20/20) | **100.0% (20/20)** | All 20 answerable queries retrieve $\ge 1$ relevant chunk within Top 2. |
| **Hit Rate@5** | 100.0% (20/20) | **100.0% (20/20)** | All 20 answerable queries retrieve $\ge 1$ relevant chunk within Top 2. |
| **Recall@1** | 90.0% (Mislabeled) | **80.0%** | Mean coverage: 15 queries @ 100%, 2 queries @ 50%, 3 queries @ 0% = 80.0%. |
| **Recall@3** | 100.0% | **100.0%** | All ground-truth chunks across all 20 queries are retrieved within Top 2. |
| **Recall@5** | 100.0% | **100.0%** | All ground-truth chunks across all 20 queries are retrieved within Top 2. |
| **MRR** | 0.9500 | **0.9250** | Based on first relevant chunk rank: $(17 \times 1.0 + 3 \times 0.5) / 20 = 0.9250$. |

---

## 5. Automated Unit Test Verification

A dedicated unit test (`test_recall_vs_hit_rate_distinction`) was added to [`tests/test_eval.py`](file:///c:/Users/Admin/OneDrive/Desktop/Alethia/tests/test_eval.py) to mathematically enforce the separation between Hit Rate@K and Recall@K.

### 5.1. Evaluation Test Suite (`pytest tests/test_eval.py -v`)
```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.2.0, pluggy-1.7.0 -- C:\Users\Admin\OneDrive\Desktop\Alethia\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Admin\OneDrive\Desktop\Alethia
plugins: anyio-4.15.1, langsmith-0.14.16
collecting ... collected 18 items

tests/test_eval.py::test_evaluation_dataset_distribution_and_integrity PASSED [  5%]
tests/test_eval.py::test_retrieval_evaluation_metrics_calculation PASSED [ 11%]
tests/test_eval.py::test_hallucination_evaluation_confusion_matrix_math PASSED [ 16%]
tests/test_eval.py::test_system_eval_abstention_and_metrics_accumulation PASSED [ 22%]
tests/test_eval.py::test_zero_rewrites_never_counts_as_self_correction PASSED [ 27%]
tests/test_eval.py::test_successful_rewrite_counted_correctly PASSED     [ 33%]
tests/test_eval.py::test_attempts_without_initial_fail_verdict_rejected PASSED [ 38%]
tests/test_eval.py::test_attempts_without_subsequent_retrieval_rejected PASSED [ 44%]
tests/test_eval.py::test_failed_rewrite_counted_correctly PASSED         [ 50%]
tests/test_eval.py::test_correct_abstention_counted_separately PASSED    [ 55%]
tests/test_eval.py::test_short_token_and_generic_word_false_positives_rejected PASSED [ 61%]
tests/test_eval.py::test_correct_evidence_chunk_and_id_recognized PASSED [ 66%]
tests/test_eval.py::test_irrelevant_chunks_rejected PASSED               [ 72%]
tests/test_eval.py::test_multiple_relevant_chunks_handled_correctly PASSED [ 77%]
tests/test_eval.py::test_unanswerable_queries_excluded_from_retrieval_recall PASSED [ 83%]
tests/test_eval.py::test_retrieval_metrics_mathematical_precision PASSED [ 88%]
tests/test_eval.py::test_retrieval_evaluation_determinism PASSED         [ 94%]
tests/test_eval.py::test_recall_vs_hit_rate_distinction PASSED           [100%]

======================== 18 passed in 77.32s (0:01:17) ========================
```

### 5.2. Full Repository Regression Test Suite (`pytest tests/ -v`)
```text
======================== 88 passed in 111.57s (0:01:51) ========================
```
- **Total Tests**: **88 passed (100%)**, 0 failures, 0 timeouts.

---

## 6. Audit Sign-Off

- **Hit Rate@K vs Recall@K**: Mathematically separated, formally defined, and verified.
- **Ground-Truth Independence**: 100% of ground-truth chunks confirmed existing in corpus and containing genuine answering evidence.
- **Metric Concordance**: Hit Rate@1 (85.0%), Hit Rate@3 (100.0%), Recall@1 (80.0%), Recall@3 (100.0%), MRR (0.9250) independently audited and serialized to JSON.
- **Documentation Synchronized**: `README.md`, `phase6_report.md`, and `eval/results/eval_benchmark_results.json` fully reconciled.
- **Architecture Integrity**: LangGraph state machine, LLM nodes, and Streamlit UI remain completely untouched.

---

## 7. Step 3 Final Evaluation Corrections & Defensible Taxonomy Realignment

Following the Step 3 independent audit, the self-correction evaluation suite (`eval/self_correction_eval.py`, `eval/results/self_correction_results.json`, `tests/test_self_correction_eval.py`) was realigned:

### 7.1. Benchmark Dataset Taxonomy Realignment (16 Cases)
- **12 Answerable Cases** (`SC-01` through `SC-11`, `SC-16`):
  - `answerability`: `"answerable"`
  - `expected_decision`: `"answer"`
  - `actual_outcome`: `"initial_success"` (dense semantic retrieval at $k=4$ retrieved target chunks at attempt 0, enabling the generator and judge to pass on cycle 1).
- **2 Out-of-Scope Cases** (`SC-12`, `SC-13`):
  - `answerability`: `"unanswerable"`
  - `expected_decision`: `"abstain"`
  - `actual_outcome`: `"abstention"` (correctly returned `"Not in context."` on attempt 0).
- **2 Corpus-Boundary Cases** (`SC-14`, `SC-15`):
  - `answerability`: `"unanswerable"`
  - `expected_decision`: `"abstain"`
  - `actual_outcome`: `"abstention"` (corrected from prior misclassification as `initial_success`).

### 7.2. Metric Denominator Realignment
- **Answer Accuracy**: $\frac{12}{12} = \mathbf{100.0\%}$ (denominator = 12 genuinely answerable cases).
- **Out-of-Scope Abstention Rate**: $\frac{2}{2} = \mathbf{100.0\%}$ (denominator = 2 out-of-scope cases).
- **Corpus-Boundary Abstention Rate**: $\frac{2}{2} = \mathbf{100.0\%}$ (denominator = 2 boundary cases).
- **Overall Abstention Rate**: $\frac{4}{16} = \mathbf{25.0\%}$ (denominator = 16 total evaluated queries; 4 abstentions).
- **Overall Decision Correctness**: $\frac{12 + 2 + 2}{16} = \frac{16}{16} = \mathbf{100.0\%}$ (strictly measured against manually defined benchmark decisions and limited mandatory-fact heuristic).
- **Cases Requiring Correction Denominator**: $\mathbf{0}$ cases.
- **Successful Correction Rate of Required**: Reported as **`N/A`** (reporting 0.0% with a denominator of 0 would be misleading).
- **Failed Correction Rate**: Reported as **`N/A`** (no cases required correction).
- **Max Rewrite Compliance Rate**: Reported as **`N/A`** (zero rewrites occurred in live benchmark run).
- **Total Live Corrections**: **0 / 16** (unexercised in live Groq run).

### 7.3. Evidence Verification Hardening & Heuristic Limitations
1. Verified strict chronological sequence: initial retrieve $\to$ initial generate $\to$ initial judge FAIL $\to$ rewrite $\to$ subsequent retrieve $\to$ revised generate $\to$ final judge PASS.
2. Verified subsequent retrieval strictly uses the exact `rewritten_question`.
3. Verified post-rewrite retrieval fetches chunk IDs that contain verified ground-truth `relevant_chunk_ids`. **Useful evidence strictly requires at least one newly retrieved relevant chunk** (`newly_retrieved_chunks = post_chunk_ids - initial_chunk_ids`, `new_relevant_chunks = newly_retrieved_chunks.intersection(target_chunks)`). Relevant chunks that were already present in attempt 0 do not count as newly retrieved evidence, and arbitrary new chunks without ground-truth relevance are strictly rejected.
4. Enforced case-specific mandatory ground-truth facts rubric (`mandatory_facts`), rejecting answers that contain only an isolated token without complete evidence.
5. **Heuristic Limitation Note**: Mandatory-fact substring matching is a limited evaluation heuristic, not a semantic correctness guarantee. While it checks for the physical presence of required factual tokens, true semantic correctness requires contextual validation, human evaluation, or advanced entailment modeling.
6. Pinned verification logic with 27 automated unit tests in `tests/test_self_correction_eval.py`.

