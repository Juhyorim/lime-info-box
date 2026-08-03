from typing import Optional
from pydantic import BaseModel

from config import DEFAULT_COLLECTION


class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    collection: str = DEFAULT_COLLECTION


class AskRequest(BaseModel):
    query: str
    top_k: int = 3
    collection: str = DEFAULT_COLLECTION
    session_id: Optional[str] = "default"
