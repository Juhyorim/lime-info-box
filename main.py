import os
import uuid
import fitz
import chromadb
from docx import Document as DocxDocument
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
# import anthropic
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware  # ← 이 줄 import에 추가

load_dotenv()

app = FastAPI(title="📚 개인 지식베이스 API")

# ← 이 블록 추가
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
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


# ── LLM Provider (Strategy 패턴) ─────────────────────────
# Spring Boot의 @Value처럼 환경변수에서 읽어옴
current_provider = os.getenv("LLM_PROVIDER", "gpt")   # gpt | claude
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


def build_rag_prompt(query: str, chunks: list, history: list) -> tuple[str, str]:
    """
    검색된 청크들을 컨텍스트로 묶어서 프롬프트 생성.
    할루시네이션 방지 지시 포함.
    """
    # 청크 → 컨텍스트 문자열
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[출처 {i+1}: {chunk['filename']} | 청크 #{chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    # 대화 히스토리 → 문자열
    history_text = ""
    if history:
        history_lines = []
        for turn in history[-6:]:   # 최근 6턴만 (토큰 절약)
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


# ── 텍스트 추출 함수 (1주차 그대로) ──────────────────────
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


# ── 대화 히스토리 (In-memory, Spring의 세션과 유사) ───────
# { session_id: [ {user: str, assistant: str}, ... ] }
conversation_store: dict[str, list] = {}


# ── API 엔드포인트 ────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 형식입니다.")

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
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {
        "status": "✅ 저장 완료", "doc_id": doc_id, "filename": file.filename,
        "file_type": ext, "total_chars": len(raw_text), "total_chunks": len(chunks),
        "preview": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/search")
async def search(req: QueryRequest):
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="저장된 문서가 없습니다.")
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
            "file_type": results["metadatas"][0][i].get("file_type", ""),
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })
    return {"query": req.query, "results": hits}


class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    session_id: Optional[str] = "default"   # 멀티턴용 세션 ID

@app.post("/ask")
async def ask(req: AskRequest):
    """
    RAG + LLM 답변 생성 엔드포인트 (2주차 핵심)
    1. ChromaDB에서 관련 청크 검색
    2. 히스토리 + 청크 → 프롬프트 생성
    3. LLM 호출 → 자연어 답변
    4. 히스토리에 저장
    """
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="저장된 문서가 없습니다.")

    # 1. 검색
    query_embedding = embed_texts([req.query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(req.top_k, collection.count()),
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

    # 2. 히스토리 불러오기
    history = conversation_store.get(req.session_id, [])

    # 3. 프롬프트 생성 + LLM 호출
    system_prompt, user_prompt = build_rag_prompt(req.query, chunks, history)
    answer = call_llm(system_prompt, user_prompt)

    # 4. 히스토리 저장
    history.append({"user": req.query, "assistant": answer})
    conversation_store[req.session_id] = history

    return {
        "query": req.query,
        "answer": answer,
        "provider": current_provider,
        "model": current_model,
        "session_id": req.session_id,
        "sources": chunks   # 출처 정보
    }


class ProviderRequest(BaseModel):
    provider: str    # gpt | claude
    model: Optional[str] = None

@app.post("/settings/provider")
async def change_provider(req: ProviderRequest):
    """
    런타임 LLM 교체 (재시작 없이 gpt ↔ claude 전환)
    """
    global current_provider, current_model
    if req.provider not in ("gpt", "claude"):
        raise HTTPException(status_code=400, detail="provider는 gpt 또는 claude만 가능합니다.")

    current_provider = req.provider

    if req.model:
        current_model = req.model
    else:
        # 기본 모델 자동 설정
        current_model = "gpt-4o-mini" if req.provider == "gpt" else "claude-haiku-4-5"

    return {
        "status": "✅ 변경 완료",
        "provider": current_provider,
        "model": current_model
    }

@app.get("/settings/provider")
async def get_provider():
    return {"provider": current_provider, "model": current_model}

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """대화 히스토리 초기화"""
    conversation_store.pop(session_id, None)
    return {"status": "✅ 대화 초기화 완료", "session_id": session_id}


@app.get("/documents")
async def list_documents():
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    doc_map = {}
    for meta in all_meta:
        doc_id = meta["doc_id"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {"doc_id": doc_id, "filename": meta["filename"], "file_type": meta.get("file_type",""), "chunk_count": 0}
        doc_map[doc_id]["chunk_count"] += 1
    return {"documents": list(doc_map.values()), "total_chunks": collection.count()}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    all_data = collection.get(include=["metadatas"])
    ids_to_delete = [all_data["ids"][i] for i, meta in enumerate(all_data["metadatas"]) if meta["doc_id"] == doc_id]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")
    collection.delete(ids=ids_to_delete)
    return {"status": "✅ 삭제 완료", "doc_id": doc_id, "deleted_chunks": len(ids_to_delete)}

@app.get("/health")
async def health():
    return {"status": "ok", "total_chunks": collection.count(), "provider": current_provider, "model": current_model}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")