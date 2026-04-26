import os
import uuid
import fitz  # PyMuPDF
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="📚 개인 지식베이스 API")

print("⏳ 임베딩 모델 로딩 중...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 임베딩 모델 로딩 완료")

# ── ChromaDB 초기화 (로컬 디스크 저장)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ════════════════════════════════════════════════
# 유틸 함수들
# ════════════════════════════════════════════════

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n[페이지 {page_num + 1}]\n{text}"
    doc.close()
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            boundary = text.rfind('\n', start, end)
            if boundary == -1:
                boundary = text.rfind('. ', start, end)
            if boundary != -1:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedder.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# ════════════════════════════════════════════════
# API 엔드포인트
# ════════════════════════════════════════════════

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    raw_text = extract_text_from_pdf(save_path)
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="텍스트를 추출할 수 없습니다. 스캔된 PDF일 수 있습니다.")

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
    embeddings = embed_texts(chunks)

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": file.filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    return {
        "status": "✅ 저장 완료",
        "doc_id": doc_id,
        "filename": file.filename,
        "total_chars": len(raw_text),
        "total_chunks": len(chunks),
        "preview": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/search")
async def search(req: QueryRequest):
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="저장된 문서가 없습니다. 먼저 PDF를 업로드하세요.")

    query_embedding = embed_texts([req.query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "rank": i + 1,
            "content": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })

    return {"query": req.query, "results": hits}


@app.get("/documents")
async def list_documents():
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    doc_map = {}
    for meta in all_meta:
        doc_id = meta["doc_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {"doc_id": doc_id, "filename": meta["filename"], "chunk_count": 0}
        doc_map[doc_id]["chunk_count"] += 1

    return {"documents": list(doc_map.values()), "total_chunks": collection.count()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    all_data = collection.get(include=["metadatas"])
    ids_to_delete = [
        all_data["ids"][i]
        for i, meta in enumerate(all_data["metadatas"])
        if meta["doc_id"] == doc_id
    ]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")

    collection.delete(ids=ids_to_delete)
    return {"status": "✅ 삭제 완료", "doc_id": doc_id, "deleted_chunks": len(ids_to_delete)}


@app.get("/health")
async def health():
    return {"status": "ok", "total_chunks": collection.count()}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")