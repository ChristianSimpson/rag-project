# data_loader.py
# --------------
# This file contains the sample documents that our RAG app will search over.
#
# In a real application, you might load documents from a database, PDF files,
# or a web API. For this project, we use a hardcoded list to keep things simple
# and focus on learning the RAG concepts rather than data management.
#
# We've chosen short paragraphs on CS and AI topics so the content is
# relevant and interesting for students learning about these technologies.


# A list of short paragraphs covering various tech topics.
# These are the "documents" that will be stored in our vector database.
SAMPLE_DOCUMENTS = [
    # Python
    "Python is a high-level programming language known for its clean, readable syntax. "
    "It uses indentation to define code blocks and emphasizes simplicity, making it one of "
    "the most beginner-friendly languages. Python is widely used in web development, data "
    "science, automation, and artificial intelligence.",

    "Python's standard library includes modules for almost everything: working with files, "
    "making HTTP requests, parsing JSON, handling dates, and much more. This 'batteries included' "
    "philosophy means you can build powerful programs without installing many extra packages.",

    # Machine Learning
    "Machine learning is a branch of artificial intelligence where systems learn from data "
    "instead of being explicitly programmed. A machine learning model finds patterns in training "
    "data and uses those patterns to make predictions or decisions on new, unseen data.",

    "Supervised learning is the most common type of machine learning. You train a model on "
    "labeled examples (input-output pairs), and it learns to predict the output for new inputs. "
    "Examples include image classification, spam detection, and house price prediction.",

    "A neural network is a machine learning model loosely inspired by the human brain. "
    "It consists of layers of interconnected nodes called neurons. Data flows through the layers, "
    "and the network learns by adjusting the strength of connections between neurons.",

    # Natural Language Processing
    "Natural Language Processing (NLP) is the field of AI focused on enabling computers to "
    "understand, interpret, and generate human language. Tasks include translation, sentiment "
    "analysis, summarization, and question answering.",

    "A transformer is a neural network architecture that revolutionized NLP. Models like GPT "
    "and BERT are built on transformers. They use a mechanism called 'attention' to weigh how "
    "important each word in a sentence is relative to every other word.",

    "Word embeddings are a way to represent words as vectors of numbers. Similar words end up "
    "close together in vector space. For example, 'king' and 'queen' would have similar vectors, "
    "as would 'dog' and 'puppy'. This lets computers understand word similarity mathematically.",

    # Databases
    "A relational database stores data in tables with rows and columns, similar to a spreadsheet. "
    "Tables are linked by relationships using foreign keys. SQL (Structured Query Language) is used "
    "to query and manipulate the data. Examples include PostgreSQL, MySQL, and SQLite.",

    "A vector database is a specialized database designed to store and search vector embeddings. "
    "Instead of exact keyword matches, it finds the most similar vectors using distance calculations. "
    "This makes it ideal for semantic search, recommendation systems, and RAG applications.",

    "NoSQL databases store data in formats other than tables — such as documents (MongoDB), "
    "key-value pairs (Redis), or graphs (Neo4j). They are often more flexible and scalable "
    "than relational databases for certain types of unstructured data.",

    # APIs & Cloud
    "An API (Application Programming Interface) is a way for two programs to communicate. "
    "A REST API uses HTTP requests (GET, POST, PUT, DELETE) to exchange data, usually in JSON "
    "format. When you use a weather app, it's calling a weather service's API behind the scenes.",

    "Cloud computing lets you use computing resources — servers, storage, databases — over the "
    "internet instead of on your own hardware. Providers like AWS, Google Cloud, and Azure offer "
    "these services on demand, so you pay only for what you use.",

    "Large Language Models (LLMs) like GPT-4 and Gemini are AI models trained on massive amounts "
    "of text data. They can generate text, answer questions, write code, and summarize documents. "
    "They're accessed through APIs and can be fine-tuned for specific tasks.",

    # RAG & AI Systems
    "Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses by first "
    "retrieving relevant documents from a knowledge base, then providing those documents as "
    "context to the LLM when generating an answer. This reduces hallucinations and grounds "
    "the model in real, verifiable information.",

    "Semantic search finds results based on meaning rather than exact keyword matches. If you "
    "search for 'fast car', semantic search can also return results about 'quick automobile' "
    "because the meanings are similar. This is powered by vector embeddings.",

    # Software Development
    "Git is a version control system that tracks changes to your code over time. It lets you "
    "save snapshots (commits) of your project, create branches for new features, and merge "
    "changes from multiple contributors. GitHub is a platform for hosting Git repositories.",

    "A virtual environment in Python is an isolated space where you can install packages "
    "without affecting other projects. This prevents conflicts between projects that need "
    "different versions of the same package. You create one with 'python -m venv venv'.",

    "Data structures are ways of organizing data so it can be accessed and modified efficiently. "
    "Common structures include lists (ordered collections), dictionaries (key-value pairs), "
    "sets (unique values), and queues (first-in, first-out ordering).",

    "Software testing is the practice of verifying that your code works correctly. Unit tests "
    "check individual functions in isolation. Integration tests check how components work "
    "together. Writing tests helps catch bugs early and makes code easier to change safely.",
]


def get_documents():
    """
    Return the list of sample documents.

    Returns:
        A list of strings, each being a document to store in the vector database.
    """
    return SAMPLE_DOCUMENTS


def generate_ids(documents):
    """
    Generate a unique ID string for each document.

    ChromaDB requires each document to have a unique string ID.
    We simply use "doc_0", "doc_1", etc.

    Args:
        documents: The list of documents to generate IDs for.

    Returns:
        A list of ID strings like ["doc_0", "doc_1", ...].
    """
    return [f"doc_{i}" for i in range(len(documents))]