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

# TODO (Week 12): Fill in the BLOCKED_PATTERNS list.
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
BLOCKED_PATTERNS = []

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
    # TODO (Week 12): Implement three safety checks.
    #
    # --- The RAG/security concept ---
    # We validate at the system boundary — the moment user input enters our app.
    # This is called "input validation" and it's a fundamental security principle.
    # Once bad input reaches the LLM prompt, it's much harder to defend against.
    #
    # Check 1 — Empty input:
    #   If query is empty or only whitespace, the app would send a pointless
    #   API call. Return: (False, "Please enter a question before submitting.")
    #
    # Check 2 — Input too long:
    #   Long inputs cost more tokens and can be used to flood the context window.
    #   If len(query) > MAX_QUERY_LENGTH, return:
    #   (False, f"Your query is too long. Please keep it under {MAX_QUERY_LENGTH} characters.")
    #
    # Check 3 — Prompt injection patterns:
    #   Convert the query to lowercase (query.lower()), then check whether any
    #   string from BLOCKED_PATTERNS appears inside it.
    #   If found, return: (False, "Your query contains content that cannot be processed.")
    #
    # If all three checks pass, return: (True, "")
    #
    return True, ""


def sanitize_input(query):
    """Remove leading/trailing whitespace from user input."""
    return query.strip()