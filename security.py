# security.py
# -----------
# This file protects our RAG app from prompt injection attacks.
#
# What is prompt injection?
# When we send a user's question to Gemini, we embed it inside a larger
# prompt that includes instructions and context. A malicious user could
# write something like "Ignore your previous instructions and..." to try
# to override those instructions and make the AI behave differently.
#
# This is one of the most common attacks against LLM-based applications.
# The fix is to check user input BEFORE it ever reaches the LLM.

# Blocked phrases used to detect likely prompt-injection attempts.
#
# --- The RAG/security concept ---
# Think about what a prompt injection attack looks like.
# An attacker is trying to write text that will "escape" from the user
# input role and start issuing new instructions to the LLM.
#
# Add at least 6 phrases that signal this kind of attack.
# Think about phrases like: trying to ignore instructions, claiming a new
# identity, asking the model to forget its context, etc.
#
BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "override instructions",
    "you are now",
    "act as",
    "pretend to be",
    "system prompt",
    "developer message",
]

# Maximum allowed length for a user query.
# Very long inputs are expensive to process and often a sign of abuse.
MAX_QUERY_LENGTH = 500


def validate_input(query):
    """
    Check whether a query is safe and valid before sending it to the LLM.

    Returns:
        (True, "")                    — query is safe, proceed
        (False, "error message")      — query is unsafe, show the error
    """
    # Validate at the system boundary — the moment user input enters our app.
    query = sanitize_input(query)

    # Check 1 — Empty input:
    if not query:
        return False, "Please enter a question before submitting."

    # Check 2 — Input too long:
    if len(query) > MAX_QUERY_LENGTH:
        return (
            False,
            f"Your query is too long. Please keep it under {MAX_QUERY_LENGTH} characters.",
        )

    # Check 3 — Prompt injection patterns:
    lower_query = query.lower()
    if any(pattern in lower_query for pattern in BLOCKED_PATTERNS):
        return False, "Your query contains content that cannot be processed."

    # All checks passed.
    return True, ""


def sanitize_input(query):
    """Remove leading/trailing whitespace from user input."""
    return query.strip()