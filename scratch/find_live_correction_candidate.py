import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import run_rag_pipeline

candidates = [
    # 1. Attention head projection and causal masking
    "What linear projection is applied after concatenating all attention heads, and why is causal masking required in autoregressive decoders?",
    
    # 2. Bias-variance tradeoff and Ridge penalty
    "Why does high variance cause overfitting on unseen test distributions, and what mathematical penalty does Ridge regression apply to model weights?",
    
    # 3. Transformer translation scores across language pairs
    "What translation BLEU scores were reported for English-to-German and English-to-French benchmarks in the Transformer paper, and what optimizer was used during training?",
    
    # 4. Planetary volcanism and satellite sizes
    "Why is Olympus Mons classified as a shield volcano, and what is its height in kilometers?",
    
    # 5. Tokenization scoring algorithms
    "How does WordPiece scoring differ from Byte-Pair Encoding, and how does SentencePiece handle whitespace?",
    
    # 6. Evaluation metrics for generation and language models
    "How is perplexity mathematically calculated for predictive distributions, and what penalty does the BLEU score apply to excessively short outputs?",
    
    # 7. Optimizer hyperparameter defaults
    "What are the specific default values for beta_1, beta_2, and epsilon in standard Adam, and how does AdamW resolve its weight decay flaw?",
    
    # 8. Gradient clipping and vanishing gradients
    "Why do deep neural networks suffer from vanishing or exploding gradients, and what maximum norm threshold is used for gradient clipping?",
    
    # 9. Cross-attention mechanism in decoders
    "In sequence-to-sequence decoders, where do the query, key, and value vectors in the cross-attention layer originate from?",
    
    # 10. Activation function comparison
    "Why does GELU smooth gradient flows compared to ReLU, and what probability distribution does it use to weight inputs?",
]

for idx, q in enumerate(candidates, 1):
    print("=" * 80)
    print(f"[{idx}] TESTING: {q}")
    try:
        state = run_rag_pipeline(q)
        att = state.get("attempts", 0)
        verdict = state.get("verdict", "")
        draft = str(state.get("draft", ""))[:100].replace("\n", " ")
        print(f"Attempts: {att} | Final Verdict: {verdict}")
        print(f"Draft: {draft}")
        
        trace = state.get("trace", [])
        judge_verdicts = [t.get("verdict") for t in trace if t.get("node") == "judge_node"]
        print(f"Judge Verdicts: {judge_verdicts}")
        
        if att > 0 and verdict == "PASS" and judge_verdicts[0] == "FAIL":
            print("\n" + "#" * 80)
            print(">>> FOUND LIVE SELF-CORRECTION CANDIDATE! <<<")
            print("#" * 80)
            break
    except Exception as e:
        print("Error:", e)

