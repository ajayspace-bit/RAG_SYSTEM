from app.services.gemini_service import embed_texts, generate_answer
from app.services.pdf_service import chunk_pages, extract_pages, find_pdf, pdf_fingerprint
from app.services.vector_service import get_collection, replace_documents, search, stored_fingerprint


def index_pdf() -> dict:
    """Extract, embed, and persist the PDF unless its content is unchanged."""
    pdf_path = find_pdf()
    fingerprint = pdf_fingerprint(pdf_path)
    if stored_fingerprint() == fingerprint:
        return {
            "message": "PDF is already indexed",
            "filename": pdf_path.name,
            "characters": sum(len(text) for _, text in extract_pages(pdf_path)),
            "chunks": get_collection().count(),
        }

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, pdf_path.name)
    embeddings = embed_texts([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")
    count = replace_documents(chunks, embeddings, fingerprint, pdf_path.name)
    return {
        "message": "PDF indexed successfully",
        "filename": pdf_path.name,
        "characters": sum(len(text) for _, text in pages),
        "chunks": count,
    }


def answer_question(question: str, top_k: int) -> dict:
    """Retrieve relevant chunks, then ask Gemini to answer from them."""
    query_embedding = embed_texts([question], "RETRIEVAL_QUERY")[0]
    results = search(query_embedding, top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    context = "\n\n".join(
        f"Retrieved Chunk {index}:\n{document}"
        for index, document in enumerate(documents, start=1)
    )
    sources = [
        {
            "chunk_id": metadata["chunk_id"],
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
        }
        for metadata, distance in zip(metadatas, distances)
    ]
    return {"answer": generate_answer(question, context), "sources": sources}