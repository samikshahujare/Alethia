"""Hallucination detection benchmark for Alethia (Phase 6).

Evaluates the LLM-as-a-Judge against controlled test pairs to quantify:
- True Positives (TP): Hallucination/contradiction correctly rejected (FAIL)
- True Negatives (TN): Supported answer or valid abstention correctly approved (PASS)
- False Positives (FP): Supported answer incorrectly rejected as hallucination
- False Negatives (FN): Hallucinated or contradictory answer erroneously approved
- Precision, Recall, and F1-score of the verification judge
"""

from typing import Any, Callable, Dict, List, Optional
from src.nodes import judge_node


HALLUCINATION_TEST_SUITE: List[Dict[str, Any]] = [
    # --- 1. Correct Supported Answers (Ground Truth: PASS / Negative) ---
    {
        "id": "HAL-01",
        "type": "correct_supported",
        "question": "How many encoder layers does the Transformer use?",
        "context": "The base architecture contains six encoder layers and six decoder layers.",
        "draft": "The base Transformer architecture contains six encoder layers.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-02",
        "type": "correct_supported",
        "question": "What is the English-to-German translation BLEU score?",
        "context": "The Transformer achieved 28.4 BLEU for English-to-German translation.",
        "draft": "The model achieved a BLEU score of 28.4 on English-to-German translation.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-03",
        "type": "correct_supported",
        "question": "What are the default beta values for Adam?",
        "context": "Standard Adam uses beta_1 equals 0.9 and beta_2 equals 0.999.",
        "draft": "Adam's default parameters are beta_1 = 0.9 and beta_2 = 0.999.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-04",
        "type": "correct_supported",
        "question": "How many natural moons does Jupiter possess?",
        "context": "Astronomical surveys confirm that Jupiter possesses 95 officially recognized natural moons.",
        "draft": "Jupiter has 95 officially recognized natural moons.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-05",
        "type": "correct_supported",
        "question": "What dataset split ratio is recommended?",
        "context": "The dataset is partitioned into an 80/10/10 ratio for training, validation, and testing.",
        "draft": "An 80/10/10 ratio is recommended for training, validation, and testing partitions.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },

    # --- 2. Contradictory Answers (Ground Truth: FAIL / Positive) ---
    {
        "id": "HAL-06",
        "type": "contradictory",
        "question": "How many encoder layers does the Transformer use?",
        "context": "The base architecture contains six encoder layers and six decoder layers.",
        "draft": "The Transformer uses twelve encoder layers and twenty-four decoder layers.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-07",
        "type": "contradictory",
        "question": "Does L1 regularization create sparse weights?",
        "context": "L1 regularization introduces a penalty driving non-essential coefficients to zero and producing sparse representations. In contrast, L2 regularization penalizes squared magnitude shrinking weights without setting them strictly to zero.",
        "draft": "L1 regularization keeps all weights non-zero, whereas L2 regularization sets non-essential weights to zero.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-08",
        "type": "contradictory",
        "question": "What is the gradient clipping threshold?",
        "context": "Gradient clipping with a maximum norm threshold of 1.0 restricts gradient magnitudes before weight updates.",
        "draft": "Gradient clipping requires a norm threshold of 15.0.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-09",
        "type": "contradictory",
        "question": "What is the height of Olympus Mons?",
        "context": "Olympus Mons is a shield volcano standing approximately 21.9 kilometers high on Mars.",
        "draft": "Olympus Mons stands approximately 8.8 kilometers high on Venus.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },

    # --- 3. Unsupported Additional Claims (Ground Truth: FAIL / Positive) ---
    {
        "id": "HAL-10",
        "type": "unsupported_claims",
        "question": "What is scaled dot-product attention?",
        "context": "Scaled dot-product attention divides dot products of queries and keys by sqrt(d_k), followed by softmax: Softmax(Q * K^T / sqrt(d_k)) * V.",
        "draft": "Scaled dot-product attention is Softmax(Q * K^T / sqrt(d_k)) * V, which was invented at Stanford University using TPU v4 chips.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-11",
        "type": "unsupported_claims",
        "question": "How does AdamW work?",
        "context": "The AdamW optimizer resolves this flaw by decoupling weight decay from gradient updates, directly decaying weights proportionally at each step.",
        "draft": "AdamW decouples weight decay and was proven to converge 500% faster than SGD on quantum computers.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-12",
        "type": "unsupported_claims",
        "question": "Why is GELU preferred over ReLU?",
        "context": "GELU weights inputs by their probability under a Gaussian cumulative distribution, which smooths gradient flows in models like BERT and GPT.",
        "draft": "GELU smooths gradient flows and eliminates 99% of all memory footprint in GPU SRAM.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-13",
        "type": "unsupported_claims",
        "question": "How many attention heads are in multi-head attention?",
        "context": "In the standard architecture, eight parallel attention heads are computed simultaneously, each with dimension d_k = 64.",
        "draft": "Eight parallel attention heads are used, with dynamic pruning introduced in 2024 to skip unneeded heads.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },

    # --- 4. Completely Fabricated Facts (Ground Truth: FAIL / Positive) ---
    {
        "id": "HAL-14",
        "type": "completely_fabricated",
        "question": "What optimizer was used to train the Transformer?",
        "context": "Training was conducted using the Adam optimizer with warmup. The Transformer achieved 28.4 BLEU.",
        "draft": "The Transformer was trained using genetic reinforcement algorithms developed by NASA.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-15",
        "type": "completely_fabricated",
        "question": "What is perplexity in language modeling?",
        "context": "Language model evaluation employs perplexity as an intrinsic quality metric. Perplexity is mathematically defined as the exponentiated average negative log-likelihood per token.",
        "draft": "Perplexity is a measurement of acoustic audio waves in speech recognition headsets.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },
    {
        "id": "HAL-16",
        "type": "completely_fabricated",
        "question": "What are Saturn's planetary rings made of?",
        "context": "Saturn is widely distinguished by its extensive and bright planetary ring system. These rings consist predominantly of billions of small water ice particles.",
        "draft": "Saturn's rings are solid titanium bands forged during prehistoric volcanic eruptions.",
        "ground_truth_verdict": "FAIL",
        "is_hallucination": True,
    },

    # --- 5. Valid Abstentions (Ground Truth: PASS / Negative) ---
    {
        "id": "HAL-17",
        "type": "valid_abstention",
        "question": "What is the capital city of Australia?",
        "context": "The Solar System contains eight designated planets. Mars hosts Olympus Mons, a shield volcano standing 21.9 km high.",
        "draft": "Not in context.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-18",
        "type": "valid_abstention",
        "question": "What is the chemical formula for caffeine?",
        "context": "The Transformer model uses multi-head attention and scaled dot-product attention.",
        "draft": "Not in context.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-19",
        "type": "valid_abstention",
        "question": "Who won the 1994 FIFA World Cup?",
        "context": "Adam optimizer uses beta_1 equals 0.9 and beta_2 equals 0.999.",
        "draft": "Not in context.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
    {
        "id": "HAL-20",
        "type": "valid_abstention",
        "question": "How many strings are on a concert harp?",
        "context": "L1 regularization drives coefficients to zero. L2 regularization penalizes squared magnitude.",
        "draft": "Not in context.",
        "ground_truth_verdict": "PASS",
        "is_hallucination": False,
    },
]


def evaluate_hallucination_detection(
    llm_func: Optional[Callable[..., str]] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute hallucination detection evaluation and calculate confusion matrix metrics.

    Definitions:
    - Positive Case: Defective answer (Ground Truth = FAIL / Hallucination)
    - Negative Case: Faithful answer or proper abstention (Ground Truth = PASS)
    - TP: Judge outputs FAIL for a defective answer (hallucination caught)
    - TN: Judge outputs PASS for a faithful answer or valid abstention
    - FP: Judge outputs FAIL for a faithful answer (false rejection)
    - FN: Judge outputs PASS for a defective answer (missed hallucination)
    """
    cases = test_cases or HALLUCINATION_TEST_SUITE

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    case_results: List[Dict[str, Any]] = []

    for item in cases:
        cid = item["id"]
        q = item["question"]
        ctx = item["context"]
        draft = item["draft"]
        gt = item["ground_truth_verdict"]

        # Run judge_node with mock chunk representing context
        state: Dict[str, Any] = {
            "question": q,
            "chunks": [{"chunk_id": f"c_{cid}", "source": "eval_test.txt", "text": ctx}],
            "draft": draft,
            "verdict": "",
            "reason": "",
            "trace": [],
        }

        judged_state = judge_node(state, llm_func=llm_func)  # type: ignore
        pred_verdict = judged_state.get("verdict", "FAIL").upper()
        reason = judged_state.get("reason", "")

        # Evaluation classification:
        if gt == "FAIL":
            if pred_verdict == "FAIL":
                tp += 1
                classification = "TP"
            else:
                fn += 1
                classification = "FN"
        else:  # gt == "PASS"
            if pred_verdict == "PASS":
                tn += 1
                classification = "TN"
            else:
                fp += 1
                classification = "FP"

        case_results.append({
            "id": cid,
            "type": item["type"],
            "ground_truth": gt,
            "prediction": pred_verdict,
            "classification": classification,
            "reason": reason,
        })

    total = len(cases)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    accuracy = round((tp + tn) / total, 4) if total > 0 else 0.0

    return {
        "total_cases": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "case_results": case_results,
    }


def format_hallucination_report(results: Dict[str, Any]) -> str:
    """Format hallucination detection evaluation into markdown."""
    lines = [
        "### Hallucination Detection & Judge Faithfulness Benchmark",
        f"- **Evaluated Test Pairs**: {results['total_cases']}",
        f"- **True Positives (TP)**: {results['true_positives']} (Hallucinations successfully rejected)",
        f"- **True Negatives (TN)**: {results['true_negatives']} (Supported answers/abstentions approved)",
        f"- **False Positives (FP)**: {results['false_positives']} (Over-rejections)",
        f"- **False Negatives (FN)**: {results['false_negatives']} (Uncaught hallucinations)",
        f"- **Precision**: {results['precision'] * 100:.1f}%",
        f"- **Recall**: {results['recall'] * 100:.1f}%",
        f"- **F1-Score**: {results['f1_score']:.4f}",
        f"- **Overall Verification Accuracy**: {results['accuracy'] * 100:.1f}%",
        "",
        "| ID | Type | Ground Truth | Judge Prediction | Outcome | Judge Assessment |",
        "|:---|:---|:---:|:---:|:---:|:---|",
    ]

    for c in results.get("case_results", []):
        outcome_badge = f"**{c['classification']}**"
        lines.append(
            f"| `{c['id']}` | {c['type']} | {c['ground_truth']} | {c['prediction']} | {outcome_badge} | {c['reason'][:50]}... |"
        )

    return "\n".join(lines)

