# rag_pipeline.py
# ---------------
# This is the heart of the RAG application.
# It orchestrates all the other modules to answer user questions.
#
# RAG stands for Retrieval-Augmented Generation:
#   1. RETRIEVAL:   Find relevant documents from our knowledge base
#   2. AUGMENTED:   Add those documents as context to our prompt
#   3. GENERATION:  Use an LLM to generate an answer based on the context
#
# This file is the central hub that grows each week:
#   Week 10: Core RAG pipeline — already complete, run it!
#   Week 11: Add conversation context    → integrate conversation.py
#   Week 12: Add input security          → integrate security.py
#   Week 13: Add hallucination monitoring → integrate monitoring.py
#   Week 14: Add filtering & fallbacks   → integrate filters.py
#   Week 15: Add query rewriting         → integrate workflow.py

import inspect

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K_RESULTS,
    TEMPERATURE,
    SIMILARITY_THRESHOLD,
)
from embeddings import embed_text, embed_documents
from vector_store import add_documents, query_similar
from data_loader import get_documents, generate_ids
from conversation import ConversationHistory
from security import validate_input, sanitize_input
from monitoring import check_hallucination, calculate_confidence
from filters import filter_by_threshold, has_relevant_results, get_fallback_response, handle_api_error
from workflow import rewrite_query

_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# WEEK 10: Core RAG — Already complete. Run the app and
# explore how these three functions work together.
# ============================================================

def initialize_vector_store():
    """
    Load all sample documents, embed them, and store them in ChromaDB.
    Called once when the app starts. After this, the vector store is ready.
    """
    documents = get_documents()
    ids = generate_ids(documents)
    embeddings = embed_documents(documents)
    add_documents(documents, embeddings, ids)
    return len(documents)


def retrieve_context(query, n_results=TOP_K_RESULTS):
    """
    Find the most relevant documents for a query using semantic search.

    How it works:
      1. The query is converted to a vector embedding
      2. ChromaDB finds the document vectors closest to the query vector
      3. "Closest" means most semantically similar — not just keyword matching

    Returns:
        (documents, distances) — matched docs and their similarity distances.
        Lower distance = more similar to the query.
    """
    query_embedding = embed_text(query)
    results = query_similar(query_embedding, n_results)
    documents = results["documents"][0]
    distances = results["distances"][0]
    return documents, distances


def generate_answer(query, context_docs, conversation_history=None):
    """
    Generate an answer using Gemini with retrieved documents as context.

    The prompt includes the retrieved documents so Gemini's answer is
    grounded in our knowledge base rather than just its training data.
    """
    context = "\n\n".join(
        [f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)]
    )

    history_section = ""
    if conversation_history is not None and conversation_history.messages:
        history_text = conversation_history.get_formatted_history()
        history_section = f"\nPrevious conversation:\n{history_text}\n"

    prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}{history_section}
Current Question: {query}

Instructions:
- Answer based primarily on the provided context documents
- If the context doesn't fully answer the question, say so clearly
- Keep your answer concise and focused
- Do not make up information that isn't in the context"""

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    return response.text


# ============================================================
# MAIN PIPELINE — run_rag()
# Each week you'll add one new block to this function.
# The Week 10 core at the bottom already works.
# ============================================================

def run_rag(query, conversation_history=None):
    """
    Run the full RAG pipeline for a user query.

    Returns a dictionary with:
      - "answer":     The generated answer string
      - "sources":    The source documents used
      - "distances":  Similarity distances for each source
      - "confidence": A 0–1 confidence score
      - "grounding":  Hallucination check result
      - "error":      Error message (empty string if no error)
    """

    is_valid, validation_error = validate_input(query)
    if not is_valid:
        return {
            "answer": validation_error,
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {},
            "error": validation_error,
        }

    query = sanitize_input(query)

    history_context = ""
    if conversation_history and conversation_history.messages:
        history_context = conversation_history.get_formatted_history()

    retrieval_query = rewrite_query(query, history_context)
    documents, distances = retrieve_context(retrieval_query)

    documents, distances = filter_by_threshold(
        documents, distances, SIMILARITY_THRESHOLD
    )

    if not has_relevant_results(documents):
        fallback = get_fallback_response()
        if conversation_history is not None:
            conversation_history.add_message("user", query)
            conversation_history.add_message("assistant", fallback)
        return {
            "answer": fallback,
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {
                "verdict": "N/A",
                "is_grounded": True,
                "warning": "",
            },
            "error": "",
        }

    try:
        answer = generate_answer(query, documents, conversation_history)
    except Exception as exc:
        err_msg = handle_api_error(exc)
        return {
            "answer": err_msg,
            "sources": [],
            "distances": [],
            "confidence": 0.0,
            "grounding": {},
            "error": err_msg,
        }

    confidence = calculate_confidence(distances)
    grounding = check_hallucination(answer, documents)

    if conversation_history is not None:
        conversation_history.add_message("user", query)
        conversation_history.add_message("assistant", answer)

    return {
        "answer": answer,
        "sources": documents,
        "distances": distances,
        "confidence": confidence,
        "grounding": grounding,
        "error": "",
    }


def get_feature_status():
    """
    Auto-detect which weekly features are implemented.

    Each check calls the student's code with a test value and sees
    whether it returns the placeholder or a real result. Used by the
    sidebar in app.py to show a live progress panel.
    """
    from conversation import ConversationHistory
    from security import BLOCKED_PATTERNS
    from monitoring import calculate_confidence
    from filters import filter_by_threshold

    # Week 11: does get_formatted_history() produce real output?
    _h = ConversationHistory()
    _h.messages = [{"role": "user", "content": "test"}]
    week11 = _h.get_formatted_history() != ""

    # Week 12: are any injection patterns defined?
    week12 = len(BLOCKED_PATTERNS) > 0

    # Week 13: does calculate_confidence() return a non-zero value?
    week13 = calculate_confidence([0.5]) != 0.0

    # Week 14: does filter_by_threshold() actually remove high-distance docs?
    _filtered, _ = filter_by_threshold(["a", "b"], [0.3, 1.5], threshold=1.0)
    week14 = len(_filtered) == 1

    week15 = "generate_content" in inspect.getsource(rewrite_query)

    return {
        "Week 11 — Conversation context": week11,
        "Week 12 — Input security": week12,
        "Week 13 — Hallucination monitoring": week13,
        "Week 14 — Filtering & fallbacks": week14,
        "Week 15 — Query rewriting": week15,
    }
