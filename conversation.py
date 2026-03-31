# conversation.py
# ---------------
# This file manages conversation history for our RAG app.
#
# Why do we need conversation history?
# LLMs have no memory between API calls — each call starts completely fresh.
# Without history, if you ask "What is Python?" and follow up with
# "Can you give me an example?", the AI has no idea what "it" refers to.
#
# The solution: before each API call, we paste the recent conversation
# directly into the prompt. The LLM "remembers" because we tell it what
# was said before. This is called "in-context memory."

from config import MAX_HISTORY_TURNS


class ConversationHistory:
    """
    Stores and manages the history of a conversation.

    Each message is a dictionary with two keys:
      - "role":    Either "user" or "assistant"
      - "content": The text of the message
    """

    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        """Add a message to the history. role is "user" or "assistant"."""
        self.messages.append({"role": role, "content": content})

    def get_formatted_history(self):
        """
        Format recent messages as a plain-text string to paste into an LLM prompt.

        This is the key RAG concept for Week 11: we turn the conversation list
        into a readable block of text that the LLM can use as context.

        Returns:
            A string like:
              User: What is Python?
              Assistant: Python is a high-level programming language...
              User: Can you give an example?
        """
        recent = self.get_recent(MAX_HISTORY_TURNS * 2)
        lines = []

        for message in recent:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")

        return "\n".join(lines)

    def get_recent(self, n):
        """Return the last n messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def clear(self):
        """Clear all messages from the history."""
        self.messages.clear()