import os

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

DEFAULT_COLLECTION = "기본"

current_provider = os.getenv("LLM_PROVIDER", "gpt")
current_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
