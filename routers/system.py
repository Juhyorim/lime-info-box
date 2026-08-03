from fastapi import APIRouter

from clients import chroma_client
from config import current_provider, current_model

router = APIRouter()


@router.get("/settings/provider")
async def get_provider():
    return {"provider": current_provider, "model": current_model}

@router.get("/health")
async def health():
    cols = chroma_client.list_collections()
    return {"status": "ok", "collections": len(cols), "provider": current_provider, "model": current_model}
