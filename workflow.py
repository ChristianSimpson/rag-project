# workflow.py
# -----------
# This file improves retrieval quality using multi-step AI workflows.
#
# The retrieval quality problem:
# The quality of a RAG answer depends heavily on what gets retrieved.
# And what gets retrieved depends on how similar the query embedding is
# to the document embeddings. If the user's query is vague or uses
# different vocabulary than the documents, retrieval suffers.
#
# Two solutions:
#
# 1. Query rewriting: Use an LLM to rewrite the user's question into a
#    version that will produce a better embedding for semantic search.
#    "tell me about that database thing" → "How do relational databases
#    store and query structured data using SQL?"
#
# 2. Query decomposition: Some questions are actually multiple questions.
#    Split them up and retrieve separately, then combine the results.
#    This is called "multi-hop retrieval."

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL
from embeddings import embed_text
from vector_store import query_similar

_client = genai.Client(api_key=GEMINI_API_KEY)


def rewrite_query(original_query, conversation_context=""):
    """
    Use Gemini to rewrite the user's query for better semantic search.

    Args:
        original_query:      The user's original question.
        conversation_context: Recent conversation history (helps resolve
                              pronouns like "it" or "that").

    Returns:
        A rewritten query string, or the original if rewriting fails.
    """
    # TODO (Week 15): Implement query rewriting using Gemini.
    #
    # --- The RAG concept ---
    # Embeddings capture meaning, but they're sensitive to phrasing.
    # A user might type casually ("how does python deal with dbs?") while
    # documents are written formally ("Python database connectivity and ORMs").
    # These two phrasings may not be close in embedding space even though
    # they mean the same thing. Query rewriting bridges that gap.
    #
    # Also important: if the user asks a follow-up like "What else can it do?",
    # the conversation_context lets you resolve "it" to the right topic.
    #
    # Steps:
    #   1. If conversation_context is not empty, include it in the prompt
    #   2. Build a prompt asking Gemini to rewrite the question to be more
    #      specific and technical, suitable for semantic search
    #   3. Call _client.models.generate_content() with temperature=0.1
    #      (low temperature = focused rewriting, not creative)
    #   4. Return response.text.strip() if it's not empty and under 500 chars
    #   5. Wrap in try/except — if anything fails, return original_query unchanged
    #
    return original_query  # placeholder — query passes through unchanged


def decompose_query(query):
    """
    Break a complex multi-part question into simpler sub-questions.

    Args:
        query: A question that may contain multiple distinct topics.

    Returns:
        A list of sub-question strings (up to 3), or [query] if it's
        already simple or if decomposition fails.
    """
    # TODO (Week 15): Implement query decomposition using Gemini.
    #
    # --- The RAG concept ---
    # Some questions have multiple parts, each requiring different documents.
    # "How does Python connect to databases, and what's the difference between
    # SQL and NoSQL?" needs documents about Python AND about SQL/NoSQL separately.
    # By splitting the question and searching for each part independently,
    # we get much better document coverage for complex questions.
    #
    # Steps:
    #   1. Build a prompt asking Gemini: if this question covers multiple topics,
    #      split it into 2-3 simpler sub-questions; otherwise return it as-is
    #   2. Call _client.models.generate_content() with temperature=0.1
    #   3. Split response.text on newlines, strip each line, drop empty/short lines
    #   4. Return at most 3 sub-questions
    #   5. Wrap in try/except — if anything fails, return [query]
    #
    return [query]  # placeholder — query is not decomposed


def multi_hop_retrieve(query, n_per_hop=2):
    """
    Retrieve documents for each sub-question and combine the results.

    Steps:
      1. Decompose the query into sub-questions
      2. Embed and search for each sub-question independently
      3. Combine results, removing duplicates

    Args:
        query:     The original complex query.
        n_per_hop: Documents to retrieve per sub-question.

    Returns:
        A deduplicated list of relevant document strings.
    """
    sub_queries = decompose_query(query)

    all_documents = []
    seen_documents = set()

    for sub_query in sub_queries:
        embedding = embed_text(sub_query)
        results = query_similar(embedding, n_per_hop)

        for doc in results["documents"][0]:
            if doc not in seen_documents:
                seen_documents.add(doc)
                all_documents.append(doc)

    return all_documents