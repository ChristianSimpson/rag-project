import logging
import os

from pydantic import BaseModel
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from google import genai  # pyright: ignore[reportMissingImports]

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not found in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)

PRIMARY_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REVIEW_MODEL_ID = os.getenv("GEMINI_REVIEW_MODEL", PRIMARY_MODEL_ID)

app = FastAPI()


class QueryRequest(BaseModel):
    question: str


def validate_user_input(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Question is too short")

    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Question is too long")


def validate_model_output(text: str):
    if text is None or text.strip() == "":
        raise HTTPException(status_code=500, detail="AI returned an empty response")

    if len(text) < 10:
        raise HTTPException(status_code=500, detail="AI response is too short")


def review_model_output(original_answer: str):
    review_prompt = f"""
You are reviewing an AI-generated response.

Your job:
- If the response is unclear, incomplete, or poorly written, improve it.
- If the response is already good, return it unchanged.

AI response to review:
{original_answer}
"""

    try:
        review_response = client.models.generate_content(
            model=REVIEW_MODEL_ID,
            contents=review_prompt,
        )
    except Exception:
        logging.exception("Gemini API error during review")
        raise HTTPException(status_code=502, detail="Upstream service error")

    reviewed = _safe_response_text(review_response)
    if reviewed is None:
        raise HTTPException(
            status_code=502,
            detail="Review step returned no text (e.g. blocked or empty response).",
        )
    return reviewed


def _safe_response_text(response):
    """Get response text; handle blocked/empty responses."""
    try:
        return response.text
    except (ValueError, AttributeError):
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-gemini")
def test_gemini():
    prompt = (
        "The company Apple launched a new phone yesterday. "
        "Key features include improved speed,a sleek design, and multiple models."
    )
    step_1_prompt = f"Step 1: Summarize the main points.\n\nText:\n{prompt}"
    try:
        response_1 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=step_1_prompt,
        )
    except Exception:
        logging.exception("Gemini API error")
        raise HTTPException(status_code=502, detail="Upstream service error")

    summary = _safe_response_text(response_1)
    if summary is None:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no text (e.g. blocked or empty response).",
        )

    step_2_prompt = f"Step 2: Extract key details from the summary.\n\nSummary:\n{summary}"
    try:
        response_2 = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=step_2_prompt,
        )
    except Exception:
        logging.exception("Gemini API error")
        raise HTTPException(status_code=502, detail="Upstream service error")

    details = _safe_response_text(response_2)
    if details is None:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no text (e.g. blocked or empty response).",
        )

    return {"prompt": prompt, "response": details}

@app.post("/query")
def query_ai(request: QueryRequest):
    validate_user_input(request.question)

    try:
        primary_response = client.models.generate_content(
            model=PRIMARY_MODEL_ID,
            contents=request.question,
        )
    except Exception:
        logging.exception("Gemini API error")
        raise HTTPException(status_code=502, detail="Upstream service error")

    raw_answer = _safe_response_text(primary_response)
    if raw_answer is None:
        raise HTTPException(
            status_code=502,
            detail="AI returned no text (e.g. blocked or empty response).",
        )

    validate_model_output(raw_answer)

    reviewed_answer = review_model_output(raw_answer)

    return {
        "question": request.question,
        "answer": reviewed_answer
    }
