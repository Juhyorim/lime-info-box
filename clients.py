import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 임베딩 모델 ──────────────────────────────────────────
print("⏳ 임베딩 모델 로딩 중...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("✅ 임베딩 모델 로딩 완료")

# ── ChromaDB ─────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ── LLM ──────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
