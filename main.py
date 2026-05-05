import os
import uuid
import fitz
import chromadb
from docx import Document as DocxDocument
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

load_dotenv()

app = FastAPI(title="📚 개인 지식베이스 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 임베딩 모델 ──────────────────────────────────────────
print("⏳ 임베딩 모델 로딩 중...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 임베딩 모델 로딩 완료")

# ── ChromaDB ─────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

DEFAULT_COLLECTION = "기본"


# ── LLM ──────────────────────────────────────────────────
current_provider = os.getenv("LLM_PROVIDER", "gpt")
current_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=0.3
    )
    return response.choices[0].message.content


# ── ChromaDB 컬렉션 헬퍼 ─────────────────────────────────
def get_collection(name: str):
    """
    컬렉션명으로 컬렉션 객체 반환.
    없으면 404 에러 (get_or_create 아님 — 명시적 생성 강제)
    """
    existing = [c.name for c in chroma_client.list_collections()]
    if name not in existing:
        raise HTTPException(status_code=404, detail=f"컬렉션 '{name}' 이 존재하지 않습니다. 먼저 생성하세요.")
    return chroma_client.get_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def get_or_create_collection(name: str):
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


# ── 텍스트 추출 ───────────────────────────────────────────
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n[페이지 {page_num + 1}]\n{text}"
    doc.close()
    return full_text

def extract_text_from_txt(file_path):
    for encoding in ["utf-8", "cp949", "euc-kr"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 인식할 수 없습니다.")

def extract_text_from_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text(file_path, extension):
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
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

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False).tolist()


# ── RAG 프롬프트 ──────────────────────────────────────────
def build_rag_prompt(query: str, chunks: list, history: list) -> tuple[str, str]:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[출처 {i+1}: {chunk['filename']} | 청크 #{chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    history_text = ""
    if history:
        history_lines = []
        for turn in history[-6:]:
            history_lines.append(f"사용자: {turn['user']}")
            history_lines.append(f"AI: {turn['assistant']}")
        history_text = "\n".join(history_lines)

    system_prompt = """당신은 개인 지식베이스 AI 비서입니다.
반드시 아래 규칙을 따르세요:
1. 반드시 제공된 컨텍스트 문서만을 근거로 답변하세요.
2. 컨텍스트에 없는 내용은 절대 추측하거나 지어내지 마세요.
3. 컨텍스트에 답이 없으면 "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 말하세요.
4. 반드시 한국어로 답변하세요.
5. 답변 마지막에 참고한 출처를 명시하세요."""

    user_prompt = f"""[이전 대화]
{history_text if history_text else "없음"}

[참고 문서]
{context}

[질문]
{query}"""

    return system_prompt, user_prompt


# ── 대화 히스토리 ─────────────────────────────────────────
conversation_store: dict[str, list] = {}


# ── 컬렉션 API ────────────────────────────────────────────

class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""

@app.post("/collections")
async def create_collection(req: CollectionCreateRequest):
    """컬렉션 생성"""
    existing = [c.name for c in chroma_client.list_collections()]
    if req.name in existing:
        raise HTTPException(status_code=409, detail=f"'{req.name}' 은 이미 존재합니다.")
    chroma_client.get_or_create_collection(
        name=req.name,
        metadata={"hnsw:space": "cosine", "description": req.description}
    )
    return {"status": "✅ 생성 완료", "name": req.name, "description": req.description}

@app.get("/collections")
async def list_collections():
    """컬렉션 목록 + 각 청크 수"""
    cols = chroma_client.list_collections()
    result = []
    for col in cols:
        c = chroma_client.get_collection(col.name)
        meta = c.metadata or {}
        result.append({
            "name": col.name,
            "description": meta.get("description", ""),
            "chunk_count": c.count()
        })
    return {"collections": result, "total": len(result)}

@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """컬렉션 삭제 (안의 문서 전부 삭제됨)"""
    existing = [c.name for c in chroma_client.list_collections()]
    if name not in existing:
        raise HTTPException(status_code=404, detail=f"'{name}' 컬렉션이 없습니다.")
    chroma_client.delete_collection(name)
    return {"status": "✅ 삭제 완료", "name": name}


# ── 업로드 API ────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection: str = Form(DEFAULT_COLLECTION)   # ← 컬렉션명 받음
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 형식입니다.")

    col = get_collection(collection)   # 없으면 404

    doc_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    raw_text = extract_text(save_path, ext)
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="텍스트를 추출할 수 없습니다.")

    chunks = chunk_text(raw_text)
    embeddings = embed_texts(chunks)

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": file.filename, "chunk_index": i, "file_type": ext} for i in range(len(chunks))]
    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {
        "status": "✅ 저장 완료", "doc_id": doc_id, "filename": file.filename,
        "collection": collection, "file_type": ext,
        "total_chars": len(raw_text), "total_chunks": len(chunks),
        "preview": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    }


# ── 검색 / 질문 API ───────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    collection: str = DEFAULT_COLLECTION

@app.post("/search")
async def search(req: QueryRequest):
    col = get_collection(req.collection)
    if col.count() == 0:
        raise HTTPException(status_code=404, detail="저장된 문서가 없습니다.")
    query_embedding = embed_texts([req.query])[0]
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, col.count()),
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
    return {"query": req.query, "collection": req.collection, "results": hits}


class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    collection: str = DEFAULT_COLLECTION
    session_id: Optional[str] = "default"

@app.post("/ask")
async def ask(req: AskRequest):
    col = get_collection(req.collection)
    if col.count() == 0:
        raise HTTPException(status_code=404, detail="저장된 문서가 없습니다.")

    query_embedding = embed_texts([req.query])[0]
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, col.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })

    # 세션 키 = session_id + collection (컬렉션별 대화 분리)
    session_key = f"{req.session_id}:{req.collection}"
    history = conversation_store.get(session_key, [])

    system_prompt, user_prompt = build_rag_prompt(req.query, chunks, history)
    answer = call_llm(system_prompt, user_prompt)

    history.append({"user": req.query, "assistant": answer})
    conversation_store[session_key] = history

    return {
        "query": req.query,
        "answer": answer,
        "collection": req.collection,
        "provider": current_provider,
        "model": current_model,
        "session_id": req.session_id,
        "sources": chunks
    }


# ── 문서 API ──────────────────────────────────────────────

@app.get("/documents")
async def list_documents(collection: str = DEFAULT_COLLECTION):
    col = get_collection(collection)
    if col.count() == 0:
        return {"documents": [], "total_chunks": 0, "collection": collection}
    all_meta = col.get(include=["metadatas"])["metadatas"]
    doc_map = {}
    for meta in all_meta:
        doc_id = meta["doc_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {"doc_id": doc_id, "filename": meta["filename"],
                               "file_type": meta.get("file_type", ""), "chunk_count": 0}
        doc_map[doc_id]["chunk_count"] += 1
    return {"documents": list(doc_map.values()), "total_chunks": col.count(), "collection": collection}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, collection: str = DEFAULT_COLLECTION):
    col = get_collection(collection)
    all_data = col.get(include=["metadatas"])
    ids_to_delete = [all_data["ids"][i] for i, meta in enumerate(all_data["metadatas"]) if meta["doc_id"] == doc_id]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")
    col.delete(ids=ids_to_delete)
    return {"status": "✅ 삭제 완료", "doc_id": doc_id, "deleted_chunks": len(ids_to_delete)}


# ── 대화 초기화 ───────────────────────────────────────────

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str, collection: str = DEFAULT_COLLECTION):
    session_key = f"{session_id}:{collection}"
    conversation_store.pop(session_key, None)
    return {"status": "✅ 대화 초기화 완료", "session_id": session_id, "collection": collection}


# ── 기타 ──────────────────────────────────────────────────

@app.get("/settings/provider")
async def get_provider():
    return {"provider": current_provider, "model": current_model}

@app.get("/health")
async def health():
    cols = chroma_client.list_collections()
    return {"status": "ok", "collections": len(cols), "provider": current_provider, "model": current_model}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")