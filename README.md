# RAG Learning App

A Retrieval-Augmented Generation (RAG) application built with Python, ChromaDB, SentenceTransformers, and Google Gemini. You'll build this incrementally over Weeks 10–15.

## What This App Does

You can ask this app questions about Python, machine learning, databases, APIs, and AI concepts. It finds the most relevant documents from its knowledge base and sends them to Gemini as context — so the answers are grounded in real information rather than guesswork.

## System Architecture

```
User Query
    │
    ▼
[security.py]      ← Validate and sanitize input (Week 12)
    │
    ▼
[workflow.py]      ← Rewrite query for better retrieval (Week 15)
    │
    ▼
[embeddings.py]    ← Convert query to a vector
    │
    ▼
[vector_store.py]  ← Find similar document vectors in ChromaDB
    │
    ▼
[filters.py]       ← Remove irrelevant results (Week 14)
    │
    ▼
[rag_pipeline.py]  ← Build prompt with retrieved context
    │
    ▼
  Gemini API       ← Generate answer
    │
    ▼
[monitoring.py]    ← Check for hallucinations (Week 13)
    │
    ▼
[app.py]           ← Display answer, sources, confidence
```

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd student-rag-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:
- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your Gemini API key

Copy the example environment file:
```bash
cp .env.example .env
```

Open `.env` and replace `your-gemini-api-key-here` with your actual key.
Get a free key at: https://aistudio.google.com/apikey

### 5. Run the app
```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `config.py` | Configuration constants |
| `embeddings.py` | Convert text to vector embeddings |
| `vector_store.py` | Store and search vectors with ChromaDB |
| `data_loader.py` | Sample tech documents |
| `rag_pipeline.py` | Central orchestration — ties everything together |
| `conversation.py` | Conversation history (Week 11) |
| `security.py` | Input validation and security (Week 12) |
| `monitoring.py` | Hallucination detection (Week 13) |
| `filters.py` | Similarity filtering and fallbacks (Week 14) |
| `workflow.py` | Query rewriting and multi-hop retrieval (Week 15) |

---

## Weekly Progress

Update this checklist as you complete each week's assignment.

- [x] Week 10 — Ran the starter app and explored the codebase
- [x] Week 11 — Implemented conversation context
- [x] Week 12 — Implemented input security
- [x] Week 13 — Implemented hallucination monitoring
- [x] Week 14 — Implemented filtering and fallbacks
- [x] Week 15 — Implemented multi-step AI workflows

---

---
## Assignment: Week 10 — Run the Starter App

**Learning objective:** Understand how a basic RAG pipeline works end-to-end.

### Background

RAG (Retrieval-Augmented Generation) connects a vector database to an LLM. Instead of asking the LLM to answer from memory (which leads to hallucination), we first *retrieve* relevant documents from our knowledge base, then *augment* the LLM's prompt with those documents so it can generate a *grounded* answer.

This week, everything is already built. Your job is to run it, understand how the pieces fit together, and answer the reflection questions below.

### What to do

1. Follow the Setup instructions above and get the app running
2. Ask the app at least 3 questions — try both on-topic and off-topic questions
3. Read through these four files and make sure you understand what each one does:
   - `data_loader.py` — where does the knowledge base come from?
   - `embeddings.py` — what does `embed_text()` return, and why?
   - `vector_store.py` — what does ChromaDB store, and how does `query_similar()` work?
   - `rag_pipeline.py` — trace a question from `run_rag()` all the way to a returned answer

### Reflection questions (be ready to discuss in class)

- What would happen if you asked a question that no document in the knowledge base covers?
- Why do we store vector embeddings instead of just the original text?
- What is the difference between keyword search and semantic search?

### ✅ When done
Check off **Week 10** in the Weekly Progress section above, then delete this entire Week 10 assignment section (from `## Assignment: Week 10` down to the next `---`).

---

---
## Assignment: Week 11 — Conversation Context

**Learning objective:** Understand how to give an LLM memory using in-context history.

### Background

LLMs have no memory between API calls. Every call starts completely fresh. This means if you ask "What is Python?" and then "Can you give an example?", the second call has no idea what "it" refers to.

The solution used in every production chatbot is simple: before each API call, paste the recent conversation history directly into the prompt. The LLM "remembers" because *we tell it* what was said before. This is called **in-context memory**.

### What to implement

**File 1 — `conversation.py`**

Implement `get_formatted_history()`. This method formats the stored messages as a plain-text block that can be pasted into a prompt. Read the TODO comment carefully — the format matters.

**File 2 — `rag_pipeline.py`**

Find the **Week 11 TODO** block inside `generate_answer()`. Replace the placeholder `history_section = ""` with logic that:
1. Checks if `conversation_history` is not None and has messages
2. Gets the formatted history with `conversation_history.get_formatted_history()`
3. Sets `history_section` to `f"\nPrevious conversation:\n{history_text}\n"`

Then find the second **Week 11 TODO** block (at the bottom of `run_rag()`). After the answer is generated, save the exchange:
```python
conversation_history.add_message("user", query)
conversation_history.add_message("assistant", answer)
```

### How to test

Run the app and try a two-part conversation:
1. Ask: *"What is machine learning?"*
2. Ask: *"What are some real-world examples of it?"*

Without your implementation, the second answer will be generic. With it, the answer will reference machine learning specifically.

### ✅ When done
Check off **Week 11** in the Weekly Progress section above, then delete this entire Week 11 assignment section.

---
---

## Week 18 — Compliance

### Trust Principles

This application hypothetically handles three categories of sensitive exposure, mapped to the following SOC 2 trust principles:

**Security** — The Gemini API key is stored in a `.env` file and never written to code or logs. Input validation in `security.py` blocks prompt injection attempts before they reach the model. These controls protect the system itself from misuse.

**Confidentiality** — User queries are treated as internal data by default, because they may contain private details (a user might paste an email address or a name into the chat box). Retrieved documents and model outputs are similarly tagged internal until proven otherwise. Confidential-tagged data is never written to plain logs.

**Privacy** — The PII detection and redaction layer in `compliance.py` ensures that personally identifiable information (emails, phone numbers, SSNs, credit card numbers, IP addresses) is masked before it appears in any log line, error message, or debug output. This protects users even if logs are accidentally exposed.

---

### Where Sensitive Data Can Appear

| Location | Risk |
|---|---|
| User chat input (`app.py` → `run_rag()`) | User may include email, phone number, or name |
| Rewritten query (`workflow.py`) | PII from original query may carry into the rewritten version |
| Retrieved documents (`vector_store.py`) | Future document sets may include PII; current set is public tech content |
| Model output (`generate_answer()`) | Gemini may echo PII back if it appeared in the prompt |
| Error messages (`filters.py → handle_api_error`) | Exception text may include query content |
| Terminal / log output | Any `print()` call could leak data if unguarded |

---

### Metadata Tag Schema

Defined in `compliance.py`. Three fields are attached to every tagged item:

| Field | Values | Meaning |
|---|---|---|
| `sensitivity` | `public` / `internal` / `confidential` / `restricted` | How carefully this data must be handled |
| `data_type` | `operational` / `pii` / `phi` / `financial` | Category of data |
| `source` | `user_input` / `document` / `model_output` / `vector_store` | Where this data originated |

A fourth field, `contains_pii` (`True`/`False`), is computed automatically by running regex patterns over the content at tag time. If PII is detected, `sensitivity` is automatically upgraded to `confidential` regardless of what was passed in.

---

### Where Redaction Occurs

Redaction is applied at every boundary where data could leave the pipeline or enter a log:

1. **Incoming user query** — `safe_log(..., force_redact=True)` in `run_rag()` before security validation
2. **Rewritten query** — `safe_log(..., force_redact=True)` before the vector store call
3. **Before sending to Gemini** — `redact_for_model()` applied to the user query and each retrieved document inside `generate_answer()`
4. **Model output** — `safe_log(..., force_redact=True)` before the result is returned
5. **Error messages** — `safe_error_log()` in the exception handler redacts before printing
6. **Document ingestion** — `tag_document()` flags any PII in the knowledge base at startup

The function `safe_log()` in `compliance.py` is a drop-in replacement for `print()`. It runs `redact_pii()` on any content before printing, so PII never appears raw in the terminal.

---

### Assumptions and Limitations

- **Regex is not perfect.** The PII patterns cover common formats (US phone numbers, standard email addresses, SSNs, credit cards, IP addresses). They will miss unusual formats and non-US phone numbers.
- **The current knowledge base contains no real PII.** The documents in `data_loader.py` are fictional tech-topic paragraphs. Tagging and redaction are implemented as production-ready infrastructure for when real documents are loaded.
- **Redaction before model does not guarantee zero leakage.** If a user describes PII in prose without matching a known pattern ("my number starts with five-five-five"), the regex will not catch it.
- **No persistent logging system is implemented.** `safe_log()` writes to stdout only. A production system would route logs to a structured logging backend with access controls.
- **Sensitivity levels are not enforced at the UI layer.** `app.py` displays answers directly. In a production app, `confidential` or `restricted` answers would require additional access checks before display.