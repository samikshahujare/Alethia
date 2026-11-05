import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever import retrieve
from src.store import get_vector_store

store = get_vector_store()
data = store.collection.get()
all_chunks = []
for i in range(len(data["ids"])):
    all_chunks.append({
        "id": data["ids"][i],
        "source": data["metadatas"][i]["source"] if data["metadatas"] else "",
        "text": data["documents"][i] if data["documents"] else "",
    })

print(f"Total indexed chunks: {len(all_chunks)}")

# Test queries where target chunk is rank 5+ initially
test_queries = [
    # Query, target_chunk_id, target_fact
    ("What specific numerical value scales the dot-product in the 2017 machine translation network?", "attention_mechanisms_txt_chunk_0001", "sqrt(d_k)"),
    ("Which celestial satellite has a physical diameter that surpasses the smallest planet in the solar system?", "solar_system_txt_chunk_0001", "Ganymede"),
    ("What is the ratio used to partition data into training, validation, and test subsets?", "machine_learning_fundamentals_txt_chunk_0005", "80/10/10"),
    ("In the paper Attention Is All You Need, what exact BLEU benchmark score was measured for English-to-French translation?", "transformer_summary_txt_chunk_0004", "41.8 BLEU"),
    ("What optimizer with decoupled weight decay was designed to fix standard Adam's flaw in transformer training?", "optimizers_guide_txt_chunk_0003", "AdamW"),
    ("What non-linear activation weights inputs by their probability under a Gaussian cumulative distribution?", "neural_network_training_txt_chunk_0002", "GELU"),
    ("What evaluation metric uses a brevity penalty to evaluate machine translation output?", "nlp_concepts_txt_chunk_0004", "BLEU"),
    ("How high is Olympus Mons on Mars compared to terrestrial mountains?", "solar_system_txt_chunk_0003", "21.9 kilometers"),
    ("What is the branching factor corresponding to the exponentiated average negative log-likelihood per token?", "nlp_concepts_txt_chunk_0003", "perplexity"),
    ("What mathematical rule does backpropagation use to compute loss derivatives with respect to weights?", "neural_network_training_txt_chunk_0000", "chain rule"),
]

for q, target_id, fact in test_queries:
    chunks = retrieve(q, k=10)
    cids = [c["chunk_id"] for c in chunks]
    rank = cids.index(target_id) + 1 if target_id in cids else -1
    print(f"Q: {q[:60]}... -> Target: {target_id} | Rank: {rank}")

