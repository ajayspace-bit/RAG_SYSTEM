from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.chat import ChatRequest
from app.services.rag_service import answer_question, index_pdf
from app.services.vector_service import get_collection


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Attempt startup indexing, while keeping API startup usable without a key."""
    try:
        index_pdf()
    except (FileNotFoundError, RuntimeError):
        pass
    yield


app = FastAPI(title="Gemini PDF RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://rag-system-five-zeta.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return "Gemini PDF RAG API is running"


@app.get("/health")
def health():
    collection = get_collection()
    metadata = collection.metadata or {}
    return {
        "status": "healthy",
        "pdf": metadata.get("source", "No PDF indexed"),
        "chunks": collection.count(),
    }


@app.post("/api/index")
def index():
    try:
        return index_pdf()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        return answer_question(request.question, request.top_k)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=502,
            detail="The AI service could not answer the question. Check the Render logs and Gemini configuration.",
        ) from error