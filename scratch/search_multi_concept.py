import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import run_rag_pipeline

candidates = [
    # Multi-fact / specific details
    "What are all four Galilean moons of Jupiter, and what is the exact height of Olympus Mons on Mars?",
    "What optimizer was used to train the base Transformer, what BLEU score was achieved on English-to-French translation, and what are Adam's default beta parameters?",
    "Which subword tokenization algorithm merges frequent pairs of symbols, and what is the exact mathematical formula for scaled dot-product attention?",
    "What regularization technique drives non-essential coefficients to zero, and what is the maximum norm threshold for gradient clipping?",
    "What is the scaling factor in scaled dot-product attention, and how many encoder and decoder layers are in the base Transformer?",
    "In the Transformer paper, what was the English-to-German BLEU score, and what optimizer was used with warmup?",
    "What is the formula for scaled dot-product attention, and what are the default beta_1 and beta_2 values in standard Adam?",
    "What is the largest moon in the solar system, and what particle sizes are found in Saturn's rings?",
    "What activation function smooths gradients in BERT and GPT, and what dataset split ratio is recommended?",
    "How does Layer Normalization differ from Batch Normalization, and what optimizer decouples weight decay?",
]

for q in candidates:
    print("=" * 70)
    print("Q:", q.encode('ascii', 'backslashreplace').decode('ascii'))
    try:
        res = run_rag_pipeline(q)
        att = res.get("attempts", 0)
        verd = res.get("verdict", "")
        print(f"Attempts: {att} | Final Verdict: {verd}")
        trace = res.get("trace", [])
        for t in trace:
            node = t.get("node", t.get("event"))
            if node == "judge_node":
                r = str(t.get("reason")).encode('ascii', 'backslashreplace').decode('ascii')
                print(f"  Judge: {t.get('verdict')} | {r}")
            elif node == "rewrite_node":
                rew = str(t.get("rewritten_question")).encode('ascii', 'backslashreplace').decode('ascii')
                print(f"  Rewrite: {rew}")
        if att > 0:
            print(">>> SUCCESSFUL REWRITE CYCLE DETECTED! <<<")
            draft = str(res.get("draft")).encode('ascii', 'backslashreplace').decode('ascii')
            print("Final Draft:", draft)
    except Exception as e:
        print("ERROR:", e)

