import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_retriever import hybrid_retrieve
from src.prompts import format_context_from_chunks, format_generate_prompt, JUDGE_SYSTEM_PROMPT, format_judge_prompt
from src.llm import call_llm

q = "What is SOP?"
chunks = hybrid_retrieve(q, k=4)
context = format_context_from_chunks(chunks)
gen_prompt = format_generate_prompt(q, context)

prompt_test = (
    "You are a strict, factual assistant. Answer using ONLY the provided context.\n"
    "Guidelines:\n"
    "1. Answer using ONLY facts directly stated in the context.\n"
    "2. If a multi-part question has some parts answered in the context and some parts missing, "
    "answer the supported parts fully and explicitly state that the missing parts are not in the context.\n"
    "3. If a term or entity is mentioned in the context (such as in projects or activities) but no formal definition is given, "
    "state what the context mentions about it and note that the document does not define the abbreviation or term.\n"
    "4. If the topic is completely absent from the context, reply 'Not in context.'\n"
    "5. Never use external knowledge, extrapolate, or invent facts."
)

draft = call_llm([
    {"role": "system", "content": prompt_test},
    {"role": "user", "content": gen_prompt}
], temperature=0.1, max_tokens=1024)

print("--- LIVE DRAFT ---")
print(draft)

judge_prompt = format_judge_prompt(q, context, draft)
judge_resp = call_llm([
    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
    {"role": "user", "content": judge_prompt}
], temperature=0.0, max_tokens=1024)

print("\n--- JUDGE EVALUATION ---")
print(judge_resp)

