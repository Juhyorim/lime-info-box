from fastapi import APIRouter, HTTPException

from clients import chroma_client
from schemas import CollectionCreateRequest
from services.collections import encode_collection_name

router = APIRouter()


@router.post("/collections")
async def create_collection(req: CollectionCreateRequest):
    existing = [c.name for c in chroma_client.list_collections()]
    encoded = encode_collection_name(req.name)

    # 동일 display_name 중복 체크
    for c in chroma_client.list_collections():
        col = chroma_client.get_collection(c.name)
        if col.metadata and col.metadata.get("display_name") == req.name:
            raise HTTPException(status_code=409, detail=f"'{req.name}' 은 이미 존재합니다.")

    chroma_client.get_or_create_collection(
        name=encoded,
        metadata={
            "hnsw:space": "cosine",
            "display_name": req.name,      # ← 원본 한글 이름 저장
            "description": req.description
        }
    )
    return {"status": "✅ 생성 완료", "name": req.name, "description": req.description}

@router.get("/collections")
async def list_collections():
    cols = chroma_client.list_collections()
    result = []
    for col in cols:
        c = chroma_client.get_collection(col.name)
        meta = c.metadata or {}
        result.append({
            "name": meta.get("display_name", col.name),   # ← 한글 이름으로 반환
            "internal_name": col.name,
            "description": meta.get("description", ""),
            "chunk_count": c.count()
        })
    return {"collections": result, "total": len(result)}

@router.delete("/collections/{name}")
async def delete_collection(name: str):
    for col in chroma_client.list_collections():
        c = chroma_client.get_collection(col.name)
        if c.metadata and c.metadata.get("display_name") == name:
            chroma_client.delete_collection(col.name)
            return {"status": "✅ 삭제 완료", "name": name}
    raise HTTPException(status_code=404, detail=f"'{name}' 컬렉션이 없습니다.")
