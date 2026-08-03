import hashlib

from fastapi import HTTPException

from clients import chroma_client


def encode_collection_name(name: str) -> str:
    """
    한글 등 특수문자 이름 → ChromaDB 호환 영문 ID로 변환
    원본 이름은 metadata에 별도 저장
    """
    hash_val = hashlib.md5(name.encode("utf-8")).hexdigest()[:12]
    return f"col_{hash_val}"


def decode_collection_name(chroma_name: str, meta: dict) -> str:
    """metadata에서 원본 이름 복원"""
    return meta.get("display_name", chroma_name)


def get_collection(name: str):
    """display_name으로 찾아서 반환"""
    for col in chroma_client.list_collections():
        c = chroma_client.get_collection(col.name)
        if c.metadata and c.metadata.get("display_name") == name:
            return c
    raise HTTPException(status_code=404, detail=f"컬렉션 '{name}' 이 존재하지 않습니다.")


def get_or_create_collection(name: str):
    encoded = encode_collection_name(name)
    return chroma_client.get_or_create_collection(
        name=encoded,
        metadata={"hnsw:space": "cosine", "display_name": name}
    )
