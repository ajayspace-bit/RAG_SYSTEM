import chromadb

from app.core.config import CHROMA_PATH, COLLECTION_NAME
from app.services.pdf_service import PdfChunk


def get_collection():
    """Open the persistent Chroma collection used by the application."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def stored_fingerprint() -> str | None:
    metadata = get_collection().metadata or {}
    return metadata.get("pdf_fingerprint")


def replace_documents(
    chunks: list[PdfChunk], embeddings: list[list[float]], fingerprint: str, source: str
) -> int:
    """Replace old vectors after a changed PDF is detected."""
    collection = get_collection()
    if collection.count():
        collection.delete(where={"indexed_document": "true"})
    if chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "indexed_document": "true",
                }
                for chunk in chunks
            ],
        )
    collection.modify(metadata={"pdf_fingerprint": fingerprint, "source": source})
    return collection.count()


def search(embedding: list[float], top_k: int):
    """Retrieve the closest chunks and their Chroma metadata."""
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )