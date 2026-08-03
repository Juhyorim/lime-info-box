import re

import fitz
from docx import Document as DocxDocument

from clients import embedder


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
            chunks.append((chunk, start))
        start = end - overlap
    return chunks


PAGE_MARKER_RE = re.compile(r"\[페이지 (\d+)\]")

def build_page_offsets(text):
    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(text)]

def page_for_offset(page_offsets, offset):
    page = None
    for marker_offset, page_num in page_offsets:
        if marker_offset > offset:
            break
        page = page_num
    return page

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False).tolist()
