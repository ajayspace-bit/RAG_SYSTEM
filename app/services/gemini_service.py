from google import genai
from google.genai import types

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_GENERATION_MODEL,
)


_gemini_client: genai.Client | None = None


def _client() -> genai.Client:
    global _gemini_client
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key":
        raise RuntimeError("Set GEMINI_API_KEY in .env before indexing or chatting.")
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Create Gemini embeddings for a list of texts."""
    response = _client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [embedding.values for embedding in response.embeddings]


def generate_answer(question: str, context: str) -> str:
    """Ask Gemini to answer strictly from retrieved PDF context."""
    prompt = f"""You are an Employee Handbook AI Assistant.

Answer the user's question using ONLY the retrieved context.

Do not use outside knowledge.

Do not invent or assume company policies.

If the answer is not available in the retrieved context, say:

\"The information is not available in the provided Employee Handbook.\"

Keep the answer clear and concise.

Retrieved Context:
{context}

User Question:
{question}
"""
    try:
        response = _client().models.generate_content(
            model=GEMINI_GENERATION_MODEL,
            contents=prompt,
        )
    except Exception as error:
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        if status_code == 429:
            raise RuntimeError(
                "Gemini API quota exceeded. Wait for the quota reset or enable billing for this project."
            ) from error
        if status_code == 404:
            raise RuntimeError(
                f"Gemini model '{GEMINI_GENERATION_MODEL}' is unavailable for this API key."
            ) from error
        if status_code == 503:
            raise RuntimeError(
                "Gemini is temporarily unavailable. Please try again shortly."
            ) from error
        raise
    return response.text or "The information is not available in the provided Employee Handbook."