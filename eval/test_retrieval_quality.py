"""Phase 2 Retrieval Quality Evaluation Benchmark.

Evaluates semantic retrieval performance across a multi-document corpus
(7 documents, ~25-35 chunks) covering diverse technical topics and unrelated content.

Measures:
- Rank of the known relevant chunk.
- Top-K retrieval precision (Recall@K / Hit@K).
- Mean Reciprocal Rank (MRR).
- Cosine similarity and distance metrics.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_documents_from_directory
from src.text_cleaner import clean_document
from src.chunker import chunk_document
from src.store import VectorStore

# Benchmark queries with ground-truth target specifications
BENCHMARK_CASES = [
    {
        "query": "How many encoder and decoder layers does the Transformer use?",
        "expected_source": "transformer_summary.txt",
        "expected_keywords": ["six encoder layers and six decoder layers"],
        "description": "Transformer layer count specification",
    },
    {
        "query": "What BLEU score was achieved for English-to-German translation?",
        "expected_source": "transformer_summary.txt",
        "expected_keywords": ["28.4 BLEU for English-to-German translation"],
        "description": "Machine translation BLEU metric",
    },
    {
        "query": "What is the mathematical formulation for scaled dot-product attention?",
        "expected_source": "attention_mechanisms.txt",
        "expected_keywords": ["Softmax(Q * K^T / sqrt(d_k)) * V", "scaling factor"],
        "description": "Attention formula & scaling factor",
    },
    {
        "query": "How many parallel attention heads are computed simultaneously in multi-head attention?",
        "expected_source": "attention_mechanisms.txt",
        "expected_keywords": ["eight parallel attention heads"],
        "description": "Multi-head count & dimension",
    },
    {
        "query": "What is the difference between L1 Lasso and L2 Ridge regularization?",
        "expected_source": "machine_learning_fundamentals.txt",
        "expected_keywords": ["L1 regularization", "sparse feature representations", "L2 regularization"],
        "description": "Regularization comparison (L1 vs L2)",
    },
    {
        "query": "What split ratio is typically used for partitioning training, validation, and test datasets?",
        "expected_source": "machine_learning_fundamentals.txt",
        "expected_keywords": ["80/10/10 ratio"],
        "description": "Dataset partition proportions",
    },
    {
        "query": "Why is GELU preferred over ReLU activation in models like BERT and GPT?",
        "expected_source": "neural_network_training.txt",
        "expected_keywords": ["GELU", "smooths gradient flows in models like BERT and GPT"],
        "description": "GELU vs ReLU activation benefits",
    },
    {
        "query": "What threshold is used for gradient clipping to prevent exploding gradients?",
        "expected_source": "neural_network_training.txt",
        "expected_keywords": ["gradient clipping", "maximum norm threshold of 1.0"],
        "description": "Gradient clipping threshold value",
    },
    {
        "query": "What default values are used for beta_1 and beta_2 hyperparameters in Adam?",
        "expected_source": "optimizers_guide.txt",
        "expected_keywords": ["beta_1 equals 0.9", "beta_2 equals 0.999"],
        "description": "Adam hyperparameter configuration",
    },
    {
        "query": "How does AdamW resolve the weight decay flaw in standard Adam?",
        "expected_source": "optimizers_guide.txt",
        "expected_keywords": ["AdamW", "decoupling weight decay"],
        "description": "AdamW decoupled weight decay mechanism",
    },
    {
        "query": "How is language model perplexity mathematically defined?",
        "expected_source": "nlp_concepts.txt",
        "expected_keywords": ["Perplexity", "exponentiated average negative log-likelihood"],
        "description": "Perplexity formula and concept",
    },
    {
        "query": "How many officially recognized natural moons does Jupiter possess?",
        "expected_source": "solar_system.txt",
        "expected_keywords": ["Jupiter", "95 officially recognized natural moons"],
        "description": "Astronomical planetary satellite count (unrelated domain)",
    },
]


def run_benchmark(k: int = 3) -> Dict[str, Any]:
    """Run full retrieval evaluation benchmark across all benchmark test cases."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Ingest all documents from data/
        data_dir = PROJECT_ROOT / "data"
        raw_docs = load_documents_from_directory(data_dir)
        if not raw_docs:
            raise RuntimeError(f"No documents found in {data_dir}")

        all_chunks = []
        for doc in raw_docs:
            cleaned = clean_document(doc)
            chunks = chunk_document(cleaned, chunk_size=400, chunk_overlap=60)
            all_chunks.extend(chunks)

        # 2. Index into an isolated VectorStore
        store = VectorStore(persist_directory=temp_dir, collection_name="benchmark_eval")
        store.index_chunks(all_chunks, overwrite=False)
        total_chunks = store.count()

        print("=" * 85)
        print("ALETHIA PHASE 2 RETRIEVAL QUALITY BENCHMARK")
        print("=" * 85)
        print(f"Corpus documents loaded: {len(raw_docs)}")
        print(f"Total chunks indexed:    {total_chunks}")
        print(f"Top-k requested:         k = {k}")
        print(f"Total benchmark queries: {len(BENCHMARK_CASES)}")
        print("-" * 85)

        hits_at_1 = 0
        hits_at_k = 0
        reciprocal_ranks: List[float] = []
        results_log: List[Dict[str, Any]] = []

        header = f"{'#':<3} | {'Query (truncated)':<38} | {'Exp Source':<20} | {'Rank':<5} | {'Sim':<6} | {'Hit@k':<5} | {'RR':<5}"
        print(header)
        print("-" * 85)

        for idx, case in enumerate(BENCHMARK_CASES, 1):
            query = case["query"]
            exp_source = case["expected_source"]
            exp_keywords = [kw.lower() for kw in case["expected_keywords"]]

            retrieved = store.retrieve_chunks(query, k=k)

            found_rank = None
            found_similarity = 0.0

            for rank, item in enumerate(retrieved, 1):
                item_text_lower = item["text"].lower()
                matches_keywords = any(kw in item_text_lower for kw in exp_keywords)
                matches_source = item["source"] == exp_source

                if matches_keywords and matches_source:
                    found_rank = rank
                    found_similarity = item["similarity"]
                    break

            if found_rank == 1:
                hits_at_1 += 1

            if found_rank is not None and found_rank <= k:
                hits_at_k += 1
                rr = 1.0 / found_rank
            else:
                rr = 0.0

            reciprocal_ranks.append(rr)

            rank_str = str(found_rank) if found_rank else "N/A"
            hit_str = "YES" if found_rank else "NO"
            sim_str = f"{found_similarity:.4f}" if found_similarity > 0 else "0.0000"
            q_trunc = query[:36] + ".." if len(query) > 38 else query

            print(f"{idx:<3} | {q_trunc:<38} | {exp_source:<20} | {rank_str:<5} | {sim_str:<6} | {hit_str:<5} | {rr:<5.2f}")

            results_log.append({
                "query": query,
                "expected_source": exp_source,
                "rank": found_rank,
                "similarity": found_similarity,
                "hit": found_rank is not None,
                "reciprocal_rank": rr,
            })

        num_queries = len(BENCHMARK_CASES)
        recall_at_1 = (hits_at_1 / num_queries) * 100.0
        recall_at_k = (hits_at_k / num_queries) * 100.0
        mrr = (sum(reciprocal_ranks) / num_queries)

        print("=" * 85)
        print("AGGREGATE RETRIEVAL METRICS:")
        print(f"Total Test Queries:        {num_queries}")
        print(f"Recall@1 (Top-1 Accuracy): {recall_at_1:.1f}% ({hits_at_1}/{num_queries})")
        print(f"Recall@{k} (Hit@{k}):          {recall_at_k:.1f}% ({hits_at_k}/{num_queries})")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print("=" * 85)

        store.close()
        return {
            "num_queries": num_queries,
            "total_chunks": total_chunks,
            "recall_at_1": recall_at_1,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "details": results_log,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    results = run_benchmark(k=3)
    if results["recall_at_k"] < 80.0:
        print("[FAIL] Retrieval quality below acceptable threshold.")
        sys.exit(1)
    else:
        print("[PASS] Retrieval quality meets benchmark standards.")
        sys.exit(0)

