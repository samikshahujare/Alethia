"""Reproducible 30-Question Evaluation Dataset for Alethia (Phase 6 / Step 2).

Covers 4 required test distributions:
- 10 Answerable factual questions (grounded in the 7-document corpus)
- 10 Unanswerable / out-of-scope questions (expected: verified abstention)
- 5 Ambiguous questions (testing self-correction query rewriting)
- 5 Retrieval challenge questions (testing semantic boundary precision and lexical traps)

Each record defines:
- id: Unique query identifier (FAC-01..10, ABS-01..10, AMB-01..05, CHL-01..05)
- category: Query category
- question: Raw query string
- expected_answer: Expected factual answer or abstention string
- relevant_source: Target document filename or None for out-of-scope
- relevant_chunk_ids: Explicit deterministic chunk IDs containing ground truth
- required_keyphrases: Multi-token keyphrases providing verified textual evidence
- supporting_facts: Filtered fact strings for answer verification (no single tokens/digits)
- expected_behavior: 'answer', 'abstain', or 'rewrite_and_answer'
"""

from typing import Any, Dict, List

EVALUATION_DATASET: List[Dict[str, Any]] = [
    # =========================================================================
    # Group 1: 10 Answerable Factual Questions
    # =========================================================================
    {
        "id": "FAC-01",
        "category": "factual",
        "question": "How many encoder and decoder layers does the base Transformer architecture use?",
        "expected_answer": "Six encoder layers and six decoder layers.",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "required_keyphrases": ["six encoder layers and six decoder layers"],
        "supporting_facts": ["six encoder", "six decoder"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-02",
        "category": "factual",
        "question": "What BLEU score did the original Transformer achieve on English-to-German translation?",
        "expected_answer": "28.4 BLEU.",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0004"],
        "required_keyphrases": ["28.4 BLEU for English-to-German translation"],
        "supporting_facts": ["28.4"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-03",
        "category": "factual",
        "question": "What is the mathematical formulation of scaled dot-product attention?",
        "expected_answer": "Softmax(Q * K^T / sqrt(d_k)) * V.",
        "relevant_source": "attention_mechanisms.txt",
        "relevant_chunk_ids": ["attention_mechanisms_txt_chunk_0001"],
        "required_keyphrases": ["Softmax(Q * K^T / sqrt(d_k)) * V", "scaled dot-product attention"],
        "supporting_facts": ["softmax", "sqrt(d_k)"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-04",
        "category": "factual",
        "question": "How many parallel attention heads are computed simultaneously in standard multi-head attention?",
        "expected_answer": "Eight parallel attention heads with dimension d_k = 64.",
        "relevant_source": "attention_mechanisms.txt",
        "relevant_chunk_ids": ["attention_mechanisms_txt_chunk_0003"],
        "required_keyphrases": ["eight parallel attention heads are computed simultaneously"],
        "supporting_facts": ["eight parallel", "d_k = 64"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-05",
        "category": "factual",
        "question": "What is the key difference in penalty formulation between L1 Lasso and L2 Ridge regularization?",
        "expected_answer": "L1 Lasso penalizes absolute values of weights producing sparsity, while L2 Ridge penalizes squared magnitudes shrinking weights toward zero.",
        "relevant_source": "machine_learning_fundamentals.txt",
        "relevant_chunk_ids": [
            "machine_learning_fundamentals_txt_chunk_0003",
            "machine_learning_fundamentals_txt_chunk_0004",
        ],
        "required_keyphrases": [
            "absolute value of the model weights",
            "squared magnitude of weights",
        ],
        "supporting_facts": ["absolute value", "squared magnitude"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-06",
        "category": "factual",
        "question": "What standard ratio is recommended for partitioning training, validation, and test datasets?",
        "expected_answer": "An 80/10/10 ratio for training, validation, and testing.",
        "relevant_source": "machine_learning_fundamentals.txt",
        "relevant_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0005"],
        "required_keyphrases": ["80/10/10 ratio for training, validation, and testing"],
        "supporting_facts": ["80/10/10"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-07",
        "category": "factual",
        "question": "Why is GELU preferred over ReLU activation in models like BERT and GPT?",
        "expected_answer": "GELU weights inputs by their probability under a Gaussian cumulative distribution, smoothing gradient flows.",
        "relevant_source": "neural_network_training.txt",
        "relevant_chunk_ids": ["neural_network_training_txt_chunk_0002"],
        "required_keyphrases": [
            "Gaussian cumulative distribution",
            "smooths gradient flows in models like BERT and GPT",
        ],
        "supporting_facts": ["gaussian cumulative", "smooths gradient"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-08",
        "category": "factual",
        "question": "What maximum norm threshold is enforced during gradient clipping to prevent exploding gradients?",
        "expected_answer": "A maximum norm threshold of 1.0.",
        "relevant_source": "neural_network_training.txt",
        "relevant_chunk_ids": ["neural_network_training_txt_chunk_0003"],
        "required_keyphrases": ["maximum norm threshold of 1.0"],
        "supporting_facts": ["threshold of 1.0", "norm threshold of 1.0"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-09",
        "category": "factual",
        "question": "What default hyperparameter values are used for beta_1 and beta_2 in the Adam optimizer?",
        "expected_answer": "beta_1 equals 0.9 and beta_2 equals 0.999.",
        "relevant_source": "optimizers_guide.txt",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0002"],
        "required_keyphrases": ["beta_1 equals 0.9", "beta_2 equals 0.999"],
        "supporting_facts": ["beta_1 equals 0.9", "0.999"],
        "expected_behavior": "answer",
    },
    {
        "id": "FAC-10",
        "category": "factual",
        "question": "How many officially recognized natural moons does Jupiter possess according to astronomical surveys?",
        "expected_answer": "95 officially recognized natural moons.",
        "relevant_source": "solar_system.txt",
        "relevant_chunk_ids": ["solar_system_txt_chunk_0001"],
        "required_keyphrases": ["95 officially recognized natural moons orbiting Jupiter"],
        "supporting_facts": ["95 officially recognized", "95 natural moons", "95 moons"],
        "expected_behavior": "answer",
    },

    # =========================================================================
    # Group 2: 10 Unanswerable / Out-of-Scope Questions
    # =========================================================================
    {
        "id": "ABS-01",
        "category": "out_of_scope",
        "question": "What is the capital city of Australia?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-02",
        "category": "out_of_scope",
        "question": "What is the molecular chemical formula for caffeine?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-03",
        "category": "out_of_scope",
        "question": "Which national team won the 1994 FIFA World Cup tournament?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-04",
        "category": "out_of_scope",
        "question": "What is the exact boiling point of liquid nitrogen expressed in Kelvin?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-05",
        "category": "out_of_scope",
        "question": "What is the maximum structural payload capacity of the Boeing 747-8F freighter?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-06",
        "category": "out_of_scope",
        "question": "How does the Mamba state-space architecture compare in memory overhead to Transformers?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-07",
        "category": "out_of_scope",
        "question": "Who was officially crowned the first emperor of the Roman Empire?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-08",
        "category": "out_of_scope",
        "question": "What programming language was used to implement the initial Linux kernel in 1991?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-09",
        "category": "out_of_scope",
        "question": "What is the approximate distance in light-years to the Andromeda Galaxy?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },
    {
        "id": "ABS-10",
        "category": "out_of_scope",
        "question": "How many strings are strung on a standard classical concert pedal harp?",
        "expected_answer": "Not in context.",
        "relevant_source": None,
        "relevant_chunk_ids": [],
        "required_keyphrases": [],
        "supporting_facts": [],
        "expected_behavior": "abstain",
    },

    # =========================================================================
    # Group 3: 5 Ambiguous Questions (Testing Query Rewriting & Self-Correction)
    # =========================================================================
    {
        "id": "AMB-01",
        "category": "ambiguous",
        "question": "How many layers?",
        "expected_answer": "Six encoder layers and six decoder layers in the Transformer base architecture.",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0002"],
        "required_keyphrases": ["six encoder layers and six decoder layers"],
        "supporting_facts": ["six encoder", "six decoder"],
        "expected_behavior": "rewrite_and_answer",
    },
    {
        "id": "AMB-02",
        "category": "ambiguous",
        "question": "What are the beta parameters?",
        "expected_answer": "In Adam: beta_1 = 0.9 and beta_2 = 0.999.",
        "relevant_source": "optimizers_guide.txt",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0002"],
        "required_keyphrases": ["beta_1 equals 0.9", "beta_2 equals 0.999"],
        "supporting_facts": ["0.9", "0.999"],
        "expected_behavior": "rewrite_and_answer",
    },
    {
        "id": "AMB-03",
        "category": "ambiguous",
        "question": "What is the formula?",
        "expected_answer": "Scaled dot-product attention formula: Softmax(Q * K^T / sqrt(d_k)) * V.",
        "relevant_source": "attention_mechanisms.txt",
        "relevant_chunk_ids": ["attention_mechanisms_txt_chunk_0001"],
        "required_keyphrases": ["Softmax(Q * K^T / sqrt(d_k)) * V", "scaled dot-product attention"],
        "supporting_facts": ["softmax", "sqrt"],
        "expected_behavior": "rewrite_and_answer",
    },
    {
        "id": "AMB-04",
        "category": "ambiguous",
        "question": "Tell me about the moons.",
        "expected_answer": "Jupiter possesses 95 moons including Galilean moons, and Ganymede is the largest moon.",
        "relevant_source": "solar_system.txt",
        "relevant_chunk_ids": [
            "solar_system_txt_chunk_0001",
            "solar_system_txt_chunk_0002",
        ],
        "required_keyphrases": [
            "95 officially recognized natural moons",
            "largest moon in the solar system",
        ],
        "supporting_facts": ["95", "ganymede", "jupiter", "galilean"],
        "expected_behavior": "rewrite_and_answer",
    },
    {
        "id": "AMB-05",
        "category": "ambiguous",
        "question": "What is the split ratio?",
        "expected_answer": "An 80/10/10 ratio for partitioning training, validation, and testing.",
        "relevant_source": "machine_learning_fundamentals.txt",
        "relevant_chunk_ids": ["machine_learning_fundamentals_txt_chunk_0005"],
        "required_keyphrases": ["80/10/10 ratio for training, validation, and testing"],
        "supporting_facts": ["80/10/10"],
        "expected_behavior": "rewrite_and_answer",
    },

    # =========================================================================
    # Group 4: 5 Retrieval Relevance Challenge Questions
    # =========================================================================
    {
        "id": "CHL-01",
        "category": "challenge",
        "question": "How does AdamW resolve the weight decay flaw present in standard Adam paired with L2 regularization?",
        "expected_answer": "AdamW decouples weight decay from gradient updates, decaying weights proportionally at each step.",
        "relevant_source": "optimizers_guide.txt",
        "relevant_chunk_ids": ["optimizers_guide_txt_chunk_0003"],
        "required_keyphrases": [
            "decoupling weight decay from gradient updates",
            "directly decaying weights proportionally at each step",
        ],
        "supporting_facts": ["decoupling weight decay", "decouple"],
        "expected_behavior": "answer",
    },
    {
        "id": "CHL-02",
        "category": "challenge",
        "question": "What mathematical metric is defined as the exponentiated average negative log-likelihood per token?",
        "expected_answer": "Perplexity is mathematically defined as the exponentiated average negative log-likelihood per token.",
        "relevant_source": "nlp_concepts.txt",
        "relevant_chunk_ids": ["nlp_concepts_txt_chunk_0003"],
        "required_keyphrases": ["exponentiated average negative log-likelihood per token"],
        "supporting_facts": ["perplexity", "negative log-likelihood"],
        "expected_behavior": "answer",
    },
    {
        "id": "CHL-03",
        "category": "challenge",
        "question": "In the original Transformer paper, what BLEU score was achieved on the WMT 2014 English-to-French benchmark?",
        "expected_answer": "41.8 BLEU for English-to-French translation.",
        "relevant_source": "transformer_summary.txt",
        "relevant_chunk_ids": ["transformer_summary_txt_chunk_0004"],
        "required_keyphrases": ["41.8 BLEU on the WMT 2014 English-to-French benchmark"],
        "supporting_facts": ["41.8"],
        "expected_behavior": "answer",
    },
    {
        "id": "CHL-04",
        "category": "challenge",
        "question": "Which specific subword tokenization algorithm treats input text as a raw stream of characters including whitespace to avoid pre-segmentation?",
        "expected_answer": "SentencePiece treats input text as a raw stream of characters, including whitespace characters.",
        "relevant_source": "nlp_concepts.txt",
        "relevant_chunk_ids": ["nlp_concepts_txt_chunk_0002"],
        "required_keyphrases": [
            "SentencePiece treats input text as a raw stream of characters",
            "raw stream of characters, including whitespace",
        ],
        "supporting_facts": ["sentencepiece", "raw stream of characters"],
        "expected_behavior": "answer",
    },
    {
        "id": "CHL-05",
        "category": "challenge",
        "question": "What is the name and height of the tallest planetary volcano documented in the solar system?",
        "expected_answer": "Olympus Mons on Mars, standing approximately 21.9 kilometers high.",
        "relevant_source": "solar_system.txt",
        "relevant_chunk_ids": ["solar_system_txt_chunk_0003"],
        "required_keyphrases": ["Olympus Mons", "approximately 21.9 kilometers high"],
        "supporting_facts": ["olympus", "21.9"],
        "expected_behavior": "answer",
    },
]


def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """Return the complete 30-case evaluation dataset."""
    return EVALUATION_DATASET
