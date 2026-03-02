# RAG Project 

This repository contains my Retrieval-Augmented Generation (RAG) project
for the GenAI Secure Coding Course

This project will be built incrementally each week. 

## Git Commands Used So Far

- git clone
- git status
- git add
- git commit 
- git push 

 # Summary of Week 4
Configured the API key as an environment variable, modifying the `.gitignore` for security, making the necessary changes to `rag_app.py` to use `os.getenv` for key loading, and illustrating execution with `uvicorn`. 

Week 5
test-gemini generates a response to prompt
Made an api call
AI API Calls are called from the backend
Rag Systems build on this exact foundation

Week 6

### Multi-step API flow

- **Step 1 – Validate input**
  - Check that the question is present, not too short, and not too long.
  - Return a clear `400` error for bad requests instead of calling the model.

- **Step 2 – Call the primary Gemini model**
  - Send the validated question to the configured Gemini model (`PRIMARY_MODEL_ID`).
  - Get the first-draft answer from the model.

- **Step 3 – Validate model output**
  - Ensure the model’s answer is non-empty and has a minimum length.
  - If the answer fails validation, return a `500` error instead of passing through unusable content.

- **Step 4 – Review with a second model**
  - Build a review prompt that includes the original answer and instructions.
  - Ask a second model (`REVIEW_MODEL_ID`) to improve clarity and completeness when needed.

### Why these steps are separated

- Each step has a **single responsibility** (validate input, generate, validate output, review).
- Failures are easier to understand: input errors vs. model errors vs. review errors.
- You can swap models, tweak validation rules, or disable the review layer without changing the other steps.

### Challenges and open questions

- Tuning thresholds: what is the “right” minimum length for answers in different domains?
- Cost vs. quality: when is the extra review pass worth the extra latency and tokens?
- Future work: adding logging and metrics around each step to better understand failures and quality over time.

Week 7
## Why these safeguards exist

### Why input validation exists
- Prevents wasting tokens/money on empty or trivial prompts.
- Reduces abuse and operational risk (very long inputs, spammy requests).
- Ensures the API returns predictable client errors (HTTP 400) instead of failing later.

### Why output validation exists
- Models can return **empty**, **blocked**, or otherwise unusable outputs.
- Catching this early lets the API return a clear server error (instead of passing bad data downstream).
- Prevents your app from treating a malformed response as a “successful” answer.

### Why a second AI model is used to review responses
- A reviewer pass can improve clarity/structure and correct obvious issues from the first draft.
- It acts as a quality-control layer before returning content to the user.
- It’s intentionally separated so you can swap reviewer behavior/models without changing the primary generation step.
