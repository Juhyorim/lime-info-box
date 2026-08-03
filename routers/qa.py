from fastapi import APIRouter, HTTPException

from config import current_provider, current_model, DEFAULT_COLLECTION
from schemas import QueryRequest, AskRequest
from services.collections import get_collection
from services.document_processing import embed_texts
from services.rag import build_rag_prompt, call_llm, conversation_store

router = APIRouter()


# ── 검색 / 질문 API ───────────────────────────────────────

@router.post("/search")
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
            "page": results["metadatas"][0][i].get("page", -1),
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })
    return {"query": req.query, "collection": req.collection, "results": hits}


@router.post("/ask")
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
            "page": results["metadatas"][0][i].get("page", -1),
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


# ── 대화 초기화 ───────────────────────────────────────────

@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str, collection: str = DEFAULT_COLLECTION):
    session_key = f"{session_id}:{collection}"
    conversation_store.pop(session_key, None)
    return {"status": "✅ 대화 초기화 완료", "session_id": session_id, "collection": collection}
