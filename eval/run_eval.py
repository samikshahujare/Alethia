"""Master Benchmark Runner for Alethia (Phase 6).

Executes comprehensive quantitative evaluation:
1. Information Retrieval Quality (Recall@1, Recall@3, Recall@5, MRR)
2. Hallucination Detection & Judge Faithfulness (TP, TN, FP, FN, Precision, Recall, F1)
3. End-to-End System Evaluation (Factual accuracy, abstention rate, self-correction, latency)
4. Saves raw results to eval/results/eval_benchmark_results.json
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from eval.dataset import EVALUATION_DATASET
from eval.hallucination_eval import (
    evaluate_hallucination_detection,
    format_hallucination_report,
)
from eval.retrieval_eval import evaluate_retrieval, format_retrieval_report
from eval.system_eval import evaluate_end_to_end_system, format_system_report
from src.config import get_groq_model, is_groq_configured
from src.store import get_vector_store


def run_full_evaluation(save_json: bool = True) -> Dict[str, Any]:
    """Execute complete benchmark suite and output comprehensive reporting."""
    print("=" * 80)
    print("ALETHIA PHASE 6: MASTER QUANTITATIVE EVALUATION BENCHMARK")
    print("=" * 80)

    results_dir = PROJECT_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "eval_benchmark_results.json"

    # Diagnostics
    configured = is_groq_configured()
    print(f"\n[Environment Status]")
    print(f"  Groq Configured: {'YES' if configured else 'NO'}")
    print(f"  Active LLM:      {get_groq_model()}")
    store = get_vector_store()
    print(f"  ChromaDB Chunks: {store.count()}")

    final_report: Dict[str, Any] = {
        "dataset_size": len(EVALUATION_DATASET),
        "groq_configured": configured,
        "model": get_groq_model(),
    }

    # 1. Retrieval Benchmark (Deterministic, 0 LLM credits)
    print("\n" + "=" * 80)
    print("1. EVALUATING RETRIEVAL QUALITY (Recall@K, MRR)")
    print("=" * 80)
    retrieval_results = evaluate_retrieval(store=store)
    final_report["retrieval"] = retrieval_results
    print(format_retrieval_report(retrieval_results))

    # 2. Hallucination Detection Benchmark
    print("\n" + "=" * 80)
    print("2. EVALUATING HALLUCINATION DETECTION (LLM-as-a-Judge)")
    print("=" * 80)
    if configured:
        hallucination_results = evaluate_hallucination_detection()
        final_report["hallucination_detection"] = hallucination_results
        print(format_hallucination_report(hallucination_results))
    else:
        print("[SKIPPED] Live Groq API key is required to evaluate LLM-as-a-Judge inference.")
        final_report["hallucination_detection"] = {"skipped": True, "reason": "No API key"}

    # 3. End-to-End Self-Correcting System Benchmark
    print("\n" + "=" * 80)
    print("3. EVALUATING END-TO-END SYSTEM & SELF-CORRECTION (30 Questions)")
    print("=" * 80)
    if configured:
        system_results = evaluate_end_to_end_system()
        final_report["system_performance"] = system_results
        print(format_system_report(system_results))
    else:
        print("[SKIPPED] Live Groq API key is required for end-to-end question answering.")
        final_report["system_performance"] = {"skipped": True, "reason": "No API key"}

    # Save complete JSON artifact
    if save_json:
        summary_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        print(f"\n[SAVED] Raw benchmark data written to: {summary_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 BENCHMARK SUITE EXECUTION COMPLETED")
    print("=" * 80)
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alethia Phase 6 Evaluation Benchmark")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON results")
    args = parser.parse_args()

    run_full_evaluation(save_json=not args.no_save)
