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

    # ── Week 11 TODO ──────────────────────────────────────────────────────────
    # Add conversation history to the prompt.
    #
    # The RAG concept: LLMs have no memory between API calls. To support
    # follow-up questions, we paste the prior conversation directly into the
    # prompt so the model can see what was already discussed.
    #
    # If conversation_history is not None and len(conversation_history) > 0:
    #   history_text = conversation_history.get_formatted_history()
    #   Set history_section to: f"\nPrevious conversation:\n{history_text}\n"
    # Otherwise set history_section = ""
    #
    # Then include {history_section} in the prompt string below (already shown).
    # ─────────────────────────────────────────────────────────────────────────
    history_section = ""  # Week 11: replace with conversation history logic

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

    # ── Week 12 TODO ──────────────────────────────────────────────────────────
    # Add input security before any processing happens.
    #
    # The RAG concept: always validate at the system boundary — the moment
    # user input enters the app, before it touches the LLM or vector store.
    # Prompt injection can hijack LLM behavior, so we stop bad input here.
    #
    # Steps:
    #   1. Call validate_input(query) → returns (is_valid, error_message)
    #   2. If not is_valid, return this dict immediately:
    #        {"answer": error_message, "sources": [], "distances": [],
    #         "confidence": 0.0, "grounding": {}, "error": error_message}
    #   3. Clean up the query: query = sanitize_input(query)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Week 15 TODO ──────────────────────────────────────────────────────────
    # Rewrite the query before retrieval to improve embedding quality.
    #
    # The RAG concept: the phrasing of the query directly affects what
    # embedding gets produced, which affects what documents get retrieved.
    # A more specific, well-formed query produces a better embedding.
    #
    # Steps:
    #   1. Get conversation context (if any):
    #        history_context = ""
    #        if conversation_history and len(conversation_history) > 0:
    #            history_context = conversation_history.get_formatted_history()
    #   2. Rewrite: query = rewrite_query(query, history_context)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Week 10: Core Retrieval — already complete ───────────────────────────
    documents, distances = retrieve_context(query)

    # ── Week 14 TODO ──────────────────────────────────────────────────────────
    # Filter out documents that aren't similar enough to be useful.
    #
    # The RAG concept: ChromaDB always returns results even when nothing is
    # relevant. Without filtering, we might generate an answer from completely
    # unrelated documents. The threshold cuts off low-quality matches.
    #
    # Steps:
    #   1. Filter: documents, distances = filter_by_threshold(documents, distances, SIMILARITY_THRESHOLD)
    #   2. If not has_relevant_results(documents), return a fallback dict:
    #        {"answer": get_fallback_response(), "sources": [], "distances": [],
    #         "confidence": 0.0,
    #         "grounding": {"verdict": "N/A", "is_grounded": True, "warning": ""},
    #         "error": ""}
    # ─────────────────────────────────────────────────────────────────────────

    # ── Week 10: Core Generation — already complete ──────────────────────────
    # Week 14: wrap this in try/except and call handle_api_error(e) on failure
    answer = generate_answer(query, documents, conversation_history)

    # ── Week 13 TODO ──────────────────────────────────────────────────────────
    # Monitor the response quality after generation.
    #
    # The RAG concept: even with context, LLMs can hallucinate. We use
    # "LLM-as-judge" — asking Gemini to evaluate its own output against the
    # source documents. We also convert vector distances into a confidence
    # score so users know how well the retrieved docs matched the query.
    #
    # Steps:
    #   1. confidence = calculate_confidence(distances)
    #   2. grounding  = check_hallucination(answer, documents)
    #   Then replace the placeholder values below with these variables.
    # ─────────────────────────────────────────────────────────────────────────
    confidence = 0.0  # Week 13: replace with calculate_confidence(distances)
    grounding = {}    # Week 13: replace with check_hallucination(answer, documents)

    # ── Week 11 TODO ──────────────────────────────────────────────────────────
    # Save this exchange to conversation history so follow-up questions work.
    #
    # The RAG concept: we store both sides of the exchange (user question AND
    # assistant answer) so get_formatted_history() can include both in the
    # next prompt. Without this step, history is never actually saved.
    #
    # Steps (only if conversation_history is not None):
    #   conversation_history.add_message("user", query)
    #   conversation_history.add_message("assistant", answer)
    # ─────────────────────────────────────────────────────────────────────────

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

    # Week 15: hard to auto-detect without an API call — check manually
    week15 = None  # None = "check manually"

    return {
        "Week 11 — Conversation context": week11,
        "Week 12 — Input security": week12,
        "Week 13 — Hallucination monitoring": week13,
        "Week 14 — Filtering & fallbacks": week14,
        "Week 15 — Query rewriting": week15,
    }
