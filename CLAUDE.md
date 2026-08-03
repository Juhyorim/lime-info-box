# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

miniInfobase is a personal knowledge-base RAG (Retrieval-Augmented Generation) service. Users upload documents (PDF/TXT/MD/DOCX), the server extracts and chunks the text, embeds it locally, stores vectors in ChromaDB, and answers natural-language questions grounded in the retrieved chunks via an LLM.

The backend is a FastAPI app split by concern (see Architecture below). `static/index.html` is a single self-contained HTML/CSS/vanilla-JS test UI served at `/` (no build step, no frontend framework).

## Commands

```bash
# Setup (Python 3.12)
python -m venv venv
source venv/bin/activate         # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the dev server
uvicorn main:app --reload --port 8000
# UI at http://localhost:8000, API docs (Swagger) at http://localhost:8000/docs
```

There is no test suite, linter, or formatter configured in this repo.

### Environment variables (`.env`)

- `OPENAI_API_KEY` — required for the `/ask` endpoint (answer generation)
- `LLM_PROVIDER` (default `gpt`) / `LLM_MODEL` (default `gpt-4o-mini`) — only the OpenAI provider is actually wired up in `call_llm()`; `anthropic` is listed but commented out in `requirements.txt` and not implemented
- Embeddings run fully locally (sentence-transformers), so no key is needed for upload/search, only for `/ask`

## Architecture

The backend follows FastAPI's standard `APIRouter` split (each router module defines `router = APIRouter()`; `main.py` wires them up with `app.include_router(...)`):

```
main.py                        # FastAPI() + CORS + include_router(...) + "/" + static mount
config.py                      # env-derived constants (UPLOAD_DIR, SUPPORTED_EXTENSIONS, DEFAULT_COLLECTION, current_provider/current_model)
clients.py                     # heavy singletons: embedder (SentenceTransformer), chroma_client, openai_client
schemas.py                     # Pydantic request models (CollectionCreateRequest, QueryRequest, AskRequest)
services/
  collections.py                 # collection name encode/decode + get_collection/get_or_create_collection
  document_processing.py         # extract_text_* , chunk_text, page-offset helpers, embed_texts
  rag.py                          # call_llm, build_rag_prompt, conversation_store
routers/
  collections.py                 # /collections (POST/GET/DELETE)
  documents.py                    # /upload, /documents (GET/DELETE)
  qa.py                            # /search, /ask, /conversation/{session_id}
  system.py                        # /settings/provider, /health
```

1. **Embedding model** (`clients.py`) — a single global `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` instance loaded at process startup (blocks server boot; loaded once, reused for every request).
2. **ChromaDB collection helpers** (`services/collections.py`) — Chroma collection names must be ASCII-safe, but the app lets users name collections in Korean/any language. `encode_collection_name()` hashes the display name (md5, first 12 chars) into the actual Chroma collection name; the human-readable name is kept in `metadata["display_name"]`. `get_collection(display_name)` linearly scans all Chroma collections to find the one whose metadata matches — there is no direct display-name → collection index. Any code working with collections must go through these helpers rather than calling `chroma_client.get_collection()` directly with a user-facing name.
3. **Text extraction** (`services/document_processing.py`) — per-extension dispatch (`extract_text`) for `.pdf` (PyMuPDF/fitz), `.txt`/`.md` (encoding fallback: utf-8 → cp949 → euc-kr), `.docx` (python-docx). PDF extraction embeds `[페이지 N]` markers in the raw text; `build_page_offsets`/`page_for_offset` map a chunk's start offset back to a page number (chunks from non-PDF sources, or before this feature existed, get `page: -1`).
4. **Chunking** (`services/document_processing.py`) — `chunk_text()` splits into ~500-char chunks with 50-char overlap, preferring to break on a newline or `". "` boundary near the chunk end rather than mid-word/mid-sentence. Returns `(chunk_text, start_offset)` pairs, not bare strings.
5. **RAG prompt** (`services/rag.py`) — `build_rag_prompt()` assembles a system prompt (Korean, instructs the model to answer only from provided context, cite sources, and say so explicitly when the answer isn't in context) plus a user prompt containing the last 6 turns of conversation history and the retrieved chunks.
6. **Conversation history** (`services/rag.py`) — `conversation_store` is an in-process `dict[str, list]` keyed by `f"{session_id}:{collection}"`, imported directly (as a shared module-level singleton) by `routers/qa.py`. This is **not persisted** — it resets on server restart and won't work across multiple worker processes.
7. **API endpoints** — see the `routers/` breakdown above; each router only imports the `services`/`config`/`clients` it needs.

### Key behaviors to preserve when modifying

- Every collection-scoped endpoint expects a `collection` display name and resolves it through `get_collection()`/`get_or_create_collection()` — never assume the display name equals the Chroma-internal name.
- `/upload` requires the target collection to already exist (`get_collection` raises 404 otherwise); it does not auto-create collections.
- `doc_id` is the first 8 chars of a UUID; chunk IDs are `f"{doc_id}_chunk_{i}"`; uploaded files are saved to `./uploads/{doc_id}_{filename}` before extraction.
- `./uploads/` and `./chroma_db/` are runtime data directories (gitignored) — don't assume they're empty or check them into version control.

## Docs

`docs/` contains reference/comparison notes the author used while choosing tools (not living documentation to keep in sync with code):
- `docs/1_문서처리관련/pdf-tools.md` — PyMuPDF vs pdfplumber vs Tesseract OCR
- `docs/2_임베딩모델관련/Embadding_models_comparison.md` — embedding model comparison (bge-m3, OpenAI, e5, Ko-SBERT); notes the current model choice trade-offs
- `docs/3_vectorDB관련/vectordb.md` — vector DB comparison (Chroma, Qdrant, Pinecone, pgvector, FAISS) and why Chroma was picked for now
