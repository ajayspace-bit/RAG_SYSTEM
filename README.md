# Gemini PDF RAG FastAPI

A small PDF question-answering application using FastAPI, `pypdf`, Gemini embeddings and generation, and persistent ChromaDB. It does not use LangChain or OpenAI.

## Installation

```bash
pip install -r requirements.txt
```

Create `.env` from the included template and set `GEMINI_API_KEY` to your Gemini API key. The other values select the Gemini models, Chroma path, collection, and PDF directory.

Put a PDF in `assets/`. The application finds the first alphabetically sorted `.pdf`; if several PDFs exist, remove the others or change `ASSETS_PATH` to a directory containing the desired PDF. This workspace's supplied `asset/` directory is also accepted as a compatibility fallback.

## Run

Run this command from the project root (`RAG`), not from inside `frontend`:

```bash
uvicorn app.main:app --reload
```

If your terminal is already inside `RAG\frontend`, use:

```bash
uvicorn app.main:app --reload --app-dir ..
```

If your terminal is in the parent `Desktop` folder, use:

```bash
uvicorn app.main:app --reload --app-dir RAG
```

Open the API documentation at http://127.0.0.1:8000/docs. To serve the plain frontend in another terminal:

From the project root:

```bash
cd frontend
python -m http.server 5500
```

Then open http://127.0.0.1:5500.

## Run with Docker

Make sure `assets/RAG_HR_Policy_20_Page.pdf` exists, then run from the project root:

```bash
docker compose up --build
```

Open the frontend at http://127.0.0.1:5500 and the API docs at http://127.0.0.1:8000/docs. The Compose setup mounts `assets/` read-only and persists ChromaDB in `chroma_db/`, so restarting containers does not lose the index. Stop the services with:

```bash
docker compose down
```

The server attempts indexing at startup and the **Index PDF** button can trigger it manually. A SHA-256 fingerprint in Chroma metadata prevents duplicate vectors. When the PDF changes, the old collection documents are replaced.

## RAG flow

```text
PDF -> pypdf -> page text -> 500-character chunks with 50-character overlap
    -> Gemini document embeddings -> persistent ChromaDB

Question -> Gemini query embedding -> ChromaDB top 3 chunks
         -> strict context-only Gemini prompt -> answer and source metadata
```

Each stored item contains its chunk text, chunk ID, source filename, and 1-based page number. ChromaDB only retrieves information; Gemini generates the final answer from the retrieved context.

## Architecture

```text
USER -> HTML/CSS/JavaScript -> FastAPI
                              |-> /api/index -> PDF -> pypdf -> chunks -> Gemini -> ChromaDB
                              `-> /api/chat -> Gemini query embedding -> ChromaDB -> context -> Gemini -> answer
```

## API

- `GET /` returns `Gemini PDF RAG API is running`.
- `GET /health` reports the indexed filename and live Chroma chunk count.
- `POST /api/index` indexes the discovered PDF and returns filename, extracted character count, and chunk count.
- `POST /api/chat` accepts `{ "question": "...", "top_k": 3 }` and returns the answer with source pages 