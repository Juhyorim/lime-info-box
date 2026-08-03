import os
import uuid

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, DEFAULT_COLLECTION
from services.collections import get_collection
from services.document_processing import (
    extract_text,
    chunk_text,
    embed_texts,
    build_page_offsets,
    page_for_offset,
)

router = APIRouter()


# ── 업로드 API ────────────────────────────────────────────

@router.post("/upload")
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

    chunk_pairs = chunk_text(raw_text)
    chunks = [c for c, _ in chunk_pairs]
    embeddings = embed_texts(chunks)

    page_offsets = build_page_offsets(raw_text) if ext == ".pdf" else []

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_index": i,
            "file_type": ext,
            "page": (page_for_offset(page_offsets, offset) or -1) if ext == ".pdf" else -1,
        }
        for i, (_, offset) in enumerate(chunk_pairs)
    ]
    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {
        "status": "✅ 저장 완료", "doc_id": doc_id, "filename": file.filename,
        "collection": collection, "file_type": ext,
        "total_chars": len(raw_text), "total_chunks": len(chunks),
        "preview": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
    }


# ── 문서 API ──────────────────────────────────────────────

@router.get("/documents")
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

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, collection: str = DEFAULT_COLLECTION):
    col = get_collection(collection)
    all_data = col.get(include=["metadatas"])
    ids_to_delete = [all_data["ids"][i] for i, meta in enumerate(all_data["metadatas"]) if meta["doc_id"] == doc_id]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="해당 문서를 찾을 수 없습니다.")
    col.delete(ids=ids_to_delete)
    return {"status": "✅ 삭제 완료", "doc_id": doc_id, "deleted_chunks": len(ids_to_delete)}
