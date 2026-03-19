# embeddings.py
# -------------
# This file handles converting text into embeddings (vectors).
#
# What is an embedding?
# An embedding is a list of numbers (a vector) that represents the
# meaning of a piece of text. Similar texts produce similar vectors,
# which is what makes semantic search possible.
#
# For example:
#   "Python is a programming language" → [0.12, -0.34, 0.87, ...]
#   "Python helps you write code"      → [0.11, -0.35, 0.85, ...]  (very similar!)
#   "I love pizza"                     → [-0.72, 0.54, -0.12, ...]  (very different!)

import importlib

from config import EMBEDDING_MODEL

# We load the model once here, at the top of the file.
# This is important — loading a model takes several seconds, so we don't
# want to reload it every time we call embed_text().
try:
    # Dynamic import avoids editor false-positives when the package isn't installed
    # in the currently selected interpreter.
    _sentence_transformers = importlib.import_module("sentence_transformers")
    SentenceTransformer = _sentence_transformers.SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'sentence-transformers'. Install it with "
        "`python3 -m pip install sentence-transformers` and make sure your IDE "
        "is using the same interpreter."
    ) from exc

_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text):
    """
    Convert a single piece of text into a vector embedding.

    Args:
        text: A string of text to embed.

    Returns:
        A list of floating point numbers (the embedding vector).
    """
    # .encode() converts text to a numpy array, then .tolist() makes it a plain Python list
    return _model.encode(text).tolist()


def embed_documents(documents):
    """
    Convert a list of documents into a list of embeddings.

    This is more efficient than calling embed_text() in a loop because
    SentenceTransformer can process multiple documents at the same time.

    Args:
        documents: A list of strings to embed.

    Returns:
        A list of embedding vectors (a list of lists of floats).
    """
    return _model.encode(documents).tolist()

