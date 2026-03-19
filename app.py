# app.py
# ------
# This is the main entry point for our RAG application.
# It creates the Streamlit web interface that users interact with.
#
# Streamlit is a Python library that turns a Python script into a web app.
# It handles all the HTML, CSS, and JavaScript for you — you just write Python.
#
# Why is the UI in a separate file?
# Keeping the UI separate from the logic makes the code easier to maintain.
# If we want to change how the app looks, we only edit app.py.
# If we want to improve the search algorithm, we edit the logic files.
# This is called "separation of concerns" — each file has one job.

import streamlit as st
from rag_pipeline import initialize_vector_store, run_rag, get_feature_status
from conversation import ConversationHistory

# --- Page Configuration ---
# This must be the FIRST Streamlit command called in the script.
st.set_page_config(
    page_title="RAG Learning App",
    page_icon="🔍",
    layout="wide",
)

# --- Initialize Session State ---
# Streamlit re-runs the entire script on every user interaction (like a button click).
# "Session state" lets us persist data between re-runs, like a conversation history.
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ConversationHistory()

if "store_initialized" not in st.session_state:
    st.session_state.store_initialized = False
    st.session_state.doc_count = 0

if "chat_messages" not in st.session_state:
    # This list stores messages for display purposes in the chat UI
    st.session_state.chat_messages = []

# --- Initialize Vector Store (once per session) ---
# We only load and embed documents on the very first run.
# After that, the vector store stays in memory for the whole session.
if not st.session_state.store_initialized:
    with st.spinner("Loading knowledge base... (this may take a moment the first time)"):
        doc_count = initialize_vector_store()
        st.session_state.store_initialized = True
        st.session_state.doc_count = doc_count

# --- Sidebar ---
with st.sidebar:
    st.title("RAG App")
    st.divider()

    st.subheader("System Info")
    st.write(f"Knowledge base: **{st.session_state.doc_count} documents**")
    st.write("Embedding model: **all-MiniLM-L6-v2**")
    st.write("LLM: **gemini-2.5-flash**")

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.conversation_history.clear()
        st.session_state.chat_messages = []
        st.rerun()

    st.divider()

    st.subheader("How It Works")
    st.markdown("""
    1. Your question is **embedded** (converted to a vector)
    2. Similar vectors are found in **ChromaDB**
    3. Relevant documents are sent to **Gemini** as context
    4. Gemini generates an answer grounded in those documents
    """)

    st.divider()

    # --- Weekly Progress Panel ---
    # This panel auto-detects which features you've implemented.
    # It updates live as you complete each week's TODOs.
    st.subheader("Your Progress")
    status = get_feature_status()
    for feature, done in status.items():
        if done is True:
            st.success(f"✅ {feature}", icon=None)
        elif done is False:
            st.warning(f"⬜ {feature} — not yet implemented", icon=None)
        else:
            # None = can't auto-detect, prompt manual check
            st.info(f"🔲 {feature} — verify manually", icon=None)

# --- Main Content ---
st.title("RAG Learning App")
st.caption("Ask questions about Python, machine learning, databases, and AI concepts")

# --- Display Chat History ---
# Loop through all previous messages and render them in chat bubbles
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # For assistant messages, show sources, confidence, and grounding info
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])} documents retrieved)"):
                for i, (source, distance) in enumerate(
                    zip(message["sources"], message["distances"])
                ):
                    # Convert distance to a 0-1 similarity score for display
                    similarity = max(0, 1 - distance / 2)
                    st.markdown(f"**Source {i+1}** — similarity: `{similarity:.2f}`")
                    st.markdown(f"> {source}")
                    st.divider()

            # Show confidence and grounding side by side
            col1, col2 = st.columns(2)
            with col1:
                confidence = message.get("confidence", 0)
                color = "green" if confidence > 0.6 else "orange" if confidence > 0.3 else "red"
                st.markdown(f"**Confidence:** :{color}[{confidence:.0%}]")
            with col2:
                grounding = message.get("grounding", {})
                verdict = grounding.get("verdict", "")
                if verdict == "GROUNDED":
                    st.markdown("**Grounding:** :green[Grounded ✓]")
                elif verdict == "PARTIAL":
                    st.markdown("**Grounding:** :orange[Partial ⚠]")
                elif verdict == "HALLUCINATED":
                    st.markdown("**Grounding:** :red[May hallucinate ✗]")

            # Show warning if grounding check raised a concern
            warning = grounding.get("warning", "")
            if warning:
                st.warning(warning)

# --- Query Input ---
# st.chat_input shows a text box pinned to the bottom of the page
query = st.chat_input("Ask a question about tech topics...")

if query:
    # Display the user's message immediately
    with st.chat_message("user"):
        st.write(query)

    # Store the user message for future re-renders
    st.session_state.chat_messages.append({"role": "user", "content": query})

    # Run the RAG pipeline and display the response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = run_rag(query, st.session_state.conversation_history)

        # Display error or answer
        if result["error"]:
            st.error(result["answer"])
        else:
            st.write(result["answer"])

        # Display sources if available
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])} documents retrieved)"):
                for i, (source, distance) in enumerate(
                    zip(result["sources"], result["distances"])
                ):
                    similarity = max(0, 1 - distance / 2)
                    st.markdown(f"**Source {i+1}** — similarity: `{similarity:.2f}`")
                    st.markdown(f"> {source}")
                    st.divider()

        # Display confidence and grounding for successful responses
        if not result["error"] and result["sources"]:
            col1, col2 = st.columns(2)
            with col1:
                confidence = result["confidence"]
                color = "green" if confidence > 0.6 else "orange" if confidence > 0.3 else "red"
                st.markdown(f"**Confidence:** :{color}[{confidence:.0%}]")
            with col2:
                grounding = result["grounding"]
                verdict = grounding.get("verdict", "")
                if verdict == "GROUNDED":
                    st.markdown("**Grounding:** :green[Grounded ✓]")
                elif verdict == "PARTIAL":
                    st.markdown("**Grounding:** :orange[Partial ⚠]")
                elif verdict == "HALLUCINATED":
                    st.markdown("**Grounding:** :red[May hallucinate ✗]")

            warning = grounding.get("warning", "")
            if warning:
                st.warning(warning)

    # Store the assistant message for future re-renders
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "distances": result["distances"],
        "confidence": result["confidence"],
        "grounding": result["grounding"],
    })