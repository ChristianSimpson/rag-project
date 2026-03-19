# vector_store.py
# ---------------
# This file manages our vector database using ChromaDB.
#
# Why do we store vectors in a database?
# When we embed all our documents, we get hundreds of vectors. We need a
# fast way to find which vectors are most similar to a query vector.
# A vector database is optimized for exactly this — it can search through
# thousands (or millions) of vectors in milliseconds using clever math.
#
# ChromaDB is an open-source vector database that runs locally,
# so we don't need to set up any servers or pay for cloud services.

import chromadb  # type: ignore[reportMissingImports]
from config import COLLECTION_NAME

# Create a ChromaDB client that stores data in memory.
# This means the data is cleared when the app restarts, but our app
# reloads the documents on startup automatically, so this works fine.
_client = chromadb.Client()


def get_or_create_collection():
    """
    Get the existing ChromaDB collection, or create it if it doesn't exist.

    A "collection" in ChromaDB is like a table in a regular database —
    it's where we store our documents and their vector embeddings together.

    Returns:
        A ChromaDB Collection object.
    """
    return _client.get_or_create_collection(name=COLLECTION_NAME)


def add_documents(documents, embeddings, ids):
    """
    Add documents and their embeddings to the vector database.

    Args:
        documents:  A list of the original text strings.
        embeddings: A list of embedding vectors (one per document).
        ids:        A list of unique ID strings (one per document).
    """
    collection = get_or_create_collection()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
    )


def query_similar(query_embedding, n_results):
    """
    Find the most similar documents to a given query embedding.

    ChromaDB compares the query vector against all stored vectors and
    returns the closest matches. "Closest" means smallest L2 distance,
    which corresponds to highest semantic similarity.

    Args:
        query_embedding: The embedding vector for the user's query.
        n_results:       How many similar documents to return.

    Returns:
        A dictionary with keys "documents", "distances", and "ids".
        Each value is a list of lists (one inner list per query submitted).
        We only submit one query at a time, so we'll access index [0].
    """
    collection = get_or_create_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results