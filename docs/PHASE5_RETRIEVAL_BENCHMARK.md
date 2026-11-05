# ALETHIA — PHASE 5 RETRIEVAL BENCHMARK AND OPTIMIZATION REPORT

## Executive Summary

Phase 5 investigated empirically whether hybrid retrieval (combining dense semantic search via ChromaDB and sparse lexical search via Okapi BM25 using Reciprocal Rank Fusion) improves retrieval quality over individual dense-only and BM25-only baselines.

The evaluation was conducted across the 30-question evaluation dataset against the 35-chunk, 7-document persistent ChromaDB corpus. All three retrieval strategies were evaluated against the exact same queries, corpus, and ground-truth relevant chunk mappings.

### Core Empirical Finding
**Hybrid retrieval achieves the highest overall retrieval quality, combining the high first-rank precision of BM25 with the robust coverage of dense search:**
- **BM25-only** achieved higher top-1 accuracy than dense-only (**90.0% vs 85.0% Hit@1**) due to exact lexical matches on queries with technical terms and numerical identifiers (e.g., `"28.4 BLEU"`, `"beta_1 = 0.9"`, `"six encoder layers"`). However, BM25 missed 1 query completely in the top 5 (**95.0% Hit@5**) on conceptual or paraphrased queries lacking exact vocabulary overlap.
- **Dense-only** achieved complete top-3 and top-5 recall (**100.0% Hit@3 and Hit@5**) through continuous semantic vector proximity, but suffered from lower top-1 rank precision (**85.0% Hit@1**).
- **Hybrid Retrieval (RRF)** successfully united both advantages: matching BM25's top-1 accuracy (**90.0% Hit@1, 85.0% Recall@1**) while maintaining Dense's 100% coverage (**100.0% Hit@3, 100.0% Hit@5**), producing the highest Mean Reciprocal Rank (**0.9417 vs 0.9250**).

---

## 1. Evaluation Methodology and Dataset Description

### Corpus & Dataset Specification
- **Corpus Size**: 7 documents, 35 chunks indexed in ChromaDB (`all-MiniLM-L6-v2`, 384 dimensions).
- **Dataset Size**: 30 queries defined in `eval/dataset.py`.
- **Query Segregation**:
  - **Answerable in-scope queries**: 20 queries (10 factual `FAC-01..10`, 5 ambiguous `AMB-01..05`, 5 challenge `CHL-01..05`) with verified ground-truth chunk IDs.
  - **Unanswerable / out-of-scope queries**: 10 queries (`ABS-01..10`) where target evidence does not exist in the corpus (expected behavior: abstain). These are strictly excluded from the retrieval denominator.
- **Ground-Truth Relevance Criteria**:
  A chunk is classified as relevant if and only if:
  1. Its `chunk_id` is explicitly listed in `relevant_chunk_ids`, OR
  2. Its source matches `relevant_source` AND its text contains verified multi-token keyphrase evidence (single tokens, digits, and generic stopwords are strictly disallowed).

### Mathematical Definitions of Metrics
Let $Q$ denote the set of answerable queries ($|Q| = N = 20$). For each query $q \in Q$:
- $\text{Rel}(q)$ is the set of ground-truth relevant chunk IDs.
- $R_K(q)$ is the ordered list of top-$K$ retrieved chunk IDs.

1. **Hit Rate@K** ($\text{Hit}@K$):
   Binary indicator of whether at least one relevant chunk is retrieved within top $K$ ranks:
   $$\text{Hit}@K(q) = \mathbb{I}(|\text{Rel}(q) \cap R_K(q)| \ge 1)$$
   $$\text{Hit Rate}@K = \frac{1}{N} \sum_{q \in Q} \text{Hit}@K(q)$$

2. **Recall@K** ($\text{Recall}@K$):
   Fraction of total ground-truth relevant chunks retrieved within top $K$ ranks:
   $$\text{Recall}@K(q) = \frac{|\text{Rel}(q) \cap R_K(q)|}{|\text{Rel}(q)|}$$
   $$\text{Recall}@K = \frac{1}{N} \sum_{q \in Q} \text{Recall}@K(q)$$

3. **Precision@K** ($\text{Precision}@K$):
   Fraction of top $K$ retrieved chunks that are relevant:
   $$\text{Precision}@K(q) = \frac{|\text{Rel}(q) \cap R_K(q)|}{K}$$
   $$\text{Precision}@K = \frac{1}{N} \sum_{q \in Q} \text{Precision}@K(q)$$
   *(Note: Because most queries have 1 target chunk, the theoretical upper bound for Precision@3 is 0.333 and for Precision@5 is 0.200).*

4. **Mean Reciprocal Rank (MRR)**:
   Reciprocal of the rank position of the first relevant chunk retrieved (or 0 if not retrieved within cutoff $K_{\max}=5$):
   $$\text{RR}(q) = \begin{cases} \frac{1}{\text{rank}_1(q)} & \text{if a relevant chunk appears at rank } \le K_{\max} \\ 0 & \text{otherwise} \end{cases}$$
   $$\text{MRR} = \frac{1}{N} \sum_{q \in Q} \text{RR}(q)$$

---

## 2. Retrieval Strategy Comparison

Evaluated across all 20 answerable queries under identical conditions ($K_{\max} = 5$):

| Metric | Dense-Only | BM25-Only | Hybrid (RRF) |
|---|:---:|:---:|:---:|
| **Hit Rate@1** | 85.0% (17/20) | **90.0%** (18/20) | **90.0%** (18/20) |
| **Hit Rate@3** | **100.0%** (20/20) | 95.0% (19/20) | **100.0%** (20/20) |
| **Hit Rate@5** | **100.0%** (20/20) | 95.0% (19/20) | **100.0%** (20/20) |
| **Recall@1** | 80.0% | **85.0%** | **85.0%** |
| **Recall@3** | **100.0%** | 95.0% | **100.0%** |
| **Recall@5** | **100.0%** | 95.0% | **100.0%** |
| **Precision@1** | 0.8500 | **0.9000** | **0.9000** |
| **Precision@3** | **0.3666** | 0.3500 | **0.3666** |
| **Precision@5** | **0.2200** | 0.2100 | **0.2200** |
| **Mean Reciprocal Rank (MRR)** | 0.9250 | 0.9250 | **0.9417** |

### Per-Query Diagnostic Insights
- **Why BM25 wins on Top-1**: On queries like `FAC-02` (*"What BLEU score did the original Transformer achieve on English-to-German translation?"*), BM25 places `transformer_summary_txt_chunk_0004` directly at rank 1 due to high TF-IDF weighting on `"BLEU"` and `"English-to-German"`. Dense retrieval ranked this at position 2 because the semantic vector was slightly pulled by general translation references.
- **Why Dense wins on Top-3/5**: On query `CHL-02` (*"What causes gradient saturation and how do we smooth its flow?"*), BM25 failed to retrieve the relevant GELU/Gaussian smoothing chunk (`neural_network_training_txt_chunk_0002`) in the top 5 because the query did not use the word `"GELU"`. Dense retrieval successfully mapped the semantic concept of saturation and smoothing to GELU at rank 2.
- **Why Hybrid wins Overall**: Reciprocal Rank Fusion ($k=60$) effectively promotes documents present in both lists while preventing catastrophic misses from single-modality blind spots, yielding the highest MRR (0.9417).

---

## 3. Parameter Grid Evaluation (Exploratory Tuning)

> **Methodological Note**: These parameter experiments represent exploratory tuning across the existing 20-query evaluation set. Because the current benchmark lacks a separate held-out query split, these results must not be interpreted as independent test validation.

A controlled grid of 36 parameter combinations was evaluated:
- **RRF Smoothing Constant ($k_{\text{rrf}}$)**: 20, 40, 60, 80
- **Weights ($w_{\text{dense}}, w_{\text{bm25}}$)**: (0.25, 0.75), (0.50, 0.50), (0.75, 0.25)
- **Candidate Pool Depth**: 10, 20, 35

### Grid Results Summary
| $k_{\text{rrf}}$ | Dense Weight | BM25 Weight | Candidate Depth | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@5 | MRR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 20 | 0.50 | 0.50 | 10 | 90.0% | 100.0% | 100.0% | 85.0% | 100.0% | **0.9417** |
| 20 | 0.50 | 0.50 | 20 | 90.0% | 100.0% | 100.0% | 85.0% | 100.0% | **0.9417** |
| 40 | 0.50 | 0.50 | 10 | 90.0% | 100.0% | 100.0% | 85.0% | 100.0% | **0.9417** |
| 60 | 0.50 | 0.50 | 20 | 90.0% | 100.0% | 100.0% | 85.0% | 100.0% | **0.9417** |
| 60 | 1.00 | 1.00 | 20 *(Default)* | 90.0% | 100.0% | 100.0% | 85.0% | 100.0% | **0.9417** |
| 20 | 0.75 | 0.25 | 20 | 85.0% | 100.0% | 100.0% | 80.0% | 100.0% | 0.9250 |
| 20 | 0.25 | 0.75 | 20 | 90.0% | 95.0% | 95.0% | 85.0% | 95.0% | 0.9250 |

### Parameter Sensitivity Analysis
1. **Weight Balance**: Balanced weights ($w_{\text{dense}} = w_{\text{bm25}} = 0.5$ or $1.0$) consistently achieve optimal MRR (0.9417). Setting dense-heavy weights ($0.75/0.25$) reduces Hit@1 to 85.0% (degrading to dense-only behavior). Setting BM25-heavy weights ($0.25/0.75$) reduces Hit@5 to 95.0% (degrading to BM25-only behavior).
2. **RRF Smoothing ($k$)**: Results are highly stable across $k \in [20, 80]$. The default $k=60$ provides smooth rank degradation without skewing toward top-ranked outliers.
3. **Candidate Depth**: Candidate pools of 10 or 20 are sufficient for the 35-chunk corpus. Depths beyond 20 yield identical top-5 rankings while increasing fusion overhead.

**Selected Configuration**:
- $k_{\text{rrf}} = 60$
- $w_{\text{dense}} = 1.0, w_{\text{bm25}} = 1.0$ (equal weighting)
- Candidate pool depth = $\max(k \times 3, 20)$

---

## 4. Question Decomposition Evaluation

Evaluated across representative single-intent and multi-intent queries:

| Query ID | Query Text | Type | Trigger Correct? | Subquestions Generated | Evidence Preserved? |
|---|---|:---:|:---:|:---:|:---:|
| `DEC-S1` | *"What is backpropagation?"* | Single | Yes (Bypass) | 1 (Original) | Yes (`chunk_0000`) |
| `DEC-S2` | *"Define precision and recall."* | Single | Yes (Bypass) | 1 (Original) | Yes (`chunk_0004`) |
| `DEC-S3` | *"Explain transformer self-attention."* | Single | Yes (Bypass) | 1 (Original) | Yes (`chunk_0000`) |
| `DEC-M1` | *"What optimizer was used in the Transformer paper, and what are Adam's default beta values?"* | Multi | Yes (Decompose) | 2 Subquestions | Yes (`trans_0004` & `opt_0002`) |
| `DEC-M2` | *"What is precision, and how is recall calculated?"* | Multi | Yes (Decompose) | 2 Subquestions | Yes (`chunk_0004`) |
| `DEC-M3` | *"What is backpropagation? And how does gradient descent update weights?"* | Multi | Yes (Decompose) | 2 Subquestions | Yes (`nn_0000` & `opt_0000`) |

### Summary Metrics
- **Trigger Accuracy**: **100.0%** (6/6). Single-intent queries bypass decomposition (0 LLM calls), while multi-intent queries trigger decomposition.
- **Evidence Preservation Rate**: **100.0%** (6/6). In all cases, relevant ground-truth chunks for all sub-intents were retained in the merged top-4 context.
- **Noise Ratio**: In multi-intent queries requiring distinct sources (e.g. `DEC-M1`), top-4 merged retrieval contained 2 relevant target chunks and 2 secondary background chunks (noise ratio: 50.0%), ensuring both independent requirements were present in context.

---

## 5. Testing and Integrity Verification

### Pytest Execution Evidence
```
pytest tests/ -q
........................................................................ [ 32%]
........................................................................ [ 64%]
........................................................................ [ 96%]
.........                                                                [100%]
225 passed in 134.12s (0:02:14)
Exit Code: 0
```
- Total tests: **225 passed** (+9 new Phase 5 tests across 13 test files).
- 0 failed, 0 skipped, 0 warnings.

### ChromaDB and Artifact Integrity Verification
- **ChromaDB Chunk Count**: Verified via direct collection API `store.collection.count()`: **Exactly 35 chunks** (unchanged).
- **Historical Artifact Preservation**: All previous benchmark artifacts in `eval/results/` remain identical in byte length and content.
- **Judge Prompts**: All 5 hardened Judge rules in `src/prompts.py` remain 100% intact.

---

## 6. Known Limitations

1. **Dataset Size Constraint**: The evaluation dataset comprises 20 answerable queries and 10 unanswerable queries against a 35-chunk corpus. While representative of the core concepts, statistical significance testing (e.g. McNemar's test or Wilcoxon signed-rank test) is underpowered on a sample of $N=20$.
2. **Corpus Saturation**: With 35 chunks, high-ranking candidates often cover the document domain broadly, limiting the divergence between candidate pool depths of 10 versus 35.
3. **Exploratory Tuning**: Parameter grid evaluation was conducted on the evaluation set itself rather than an independent held-out split; production deployment should validate these parameters on external unseen domains.

---

## 7. Reproduction Commands

To reproduce all Phase 5 benchmark comparisons, parameter sweeps, and unit tests:

```bash
# 1. Run all unit and integration tests
.venv/Scripts/pytest tests/ -q

# 2. Run Phase 5 benchmark and generate JSON artifacts
.venv/Scripts/python -m eval.phase5_benchmark
```

