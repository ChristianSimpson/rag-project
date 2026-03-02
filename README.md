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
