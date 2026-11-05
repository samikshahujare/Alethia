"""Alethia - Self-Correcting RAG Engine."""

from src.config import validate_groq_config, is_groq_configured, get_groq_api_key
from src.document_loader import Document, load_document, load_documents_from_directory
from src.text_cleaner import clean_text, clean_document
from src.chunker import TextChunk, chunk_text, chunk_document
from src.store import VectorStore, get_vector_store, retrieve_chunks
from src.llm import call_llm, get_groq_client, DEFAULT_MODEL
from src.state import GraphState, create_initial_state
from src.nodes import retrieve_node, generate_node, judge_node, rewrite_node
from src.graph import (
    build_rag_graph,
    get_compiled_graph,
    run_rag_pipeline,
    route_after_judge,
    MAX_REWRITES,
)

__all__ = [
    "validate_groq_config",
    "is_groq_configured",
    "get_groq_api_key",
    "Document",
    "load_document",
    "load_documents_from_directory",
    "clean_text",
    "clean_document",
    "TextChunk",
    "chunk_text",
    "chunk_document",
    "VectorStore",
    "get_vector_store",
    "retrieve_chunks",
    "call_llm",
    "get_groq_client",
    "DEFAULT_MODEL",
    "GraphState",
    "create_initial_state",
    "retrieve_node",
    "generate_node",
    "judge_node",
    "rewrite_node",
    "build_rag_graph",
    "get_compiled_graph",
    "run_rag_pipeline",
    "route_after_judge",
    "MAX_REWRITES",
]
__version__ = "1.0.0"
