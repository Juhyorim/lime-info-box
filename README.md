# 📚 miniInfobase

개인 문서(PDF)를 업로드하고 자연어로 검색할 수 있는 **개인 지식베이스 AI 비서**입니다.

<br/><br/>

## ✨ 주요 기능

- **PDF 업로드** — PDF 파일을 업로드하면 자동으로 텍스트를 추출하고 벡터DB에 저장
- **자연어 검색** — 질문을 입력하면 관련된 문서 청크를 유사도 순으로 반환
- **문서 관리** — 저장된 문서 목록 확인 및 삭제

<br/><br/>

## 🛠️ 기술 스택

| 역할            | 기술                                                          |
| --------------- | ------------------------------------------------------------- |
| 웹 프레임워크   | FastAPI                                                       |
| PDF 텍스트 추출 | PyMuPDF                                                       |
| 임베딩 모델     | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| 벡터 DB         | ChromaDB                                                      |
| 서버            | Uvicorn                                                       |

<br/><br/>

## ⚙️ 동작 흐름

```
PDF 업로드
    ↓
텍스트 추출 (PyMuPDF)
    ↓
청크 분할 (500자 단위, 50자 오버랩)
    ↓
임베딩 생성 (sentence-transformers)
    ↓
ChromaDB 저장
    ↓
질문 입력 → 임베딩 변환 → 코사인 유사도 검색 → 결과 반환
```

<br/><br/>

## 🚀 실행 방법

### 사전 요구사항

- Python 3.12
- Microsoft C++ Build Tools (ChromaDB 빌드에 필요)

### 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/miniInfobase.git
cd miniInfobase

# 2. 가상환경 생성 및 활성화
py -3.12 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 서버 실행
uvicorn main:app --reload --port 8000
```

### 접속

```
http://localhost:8000
```

<br/><br/>

## 📁 프로젝트 구조

```
miniInfobase/
├── main.py              # FastAPI 백엔드
├── requirements.txt     # 의존성 목록
├── static/
│   └── index.html       # 테스트용 UI
├── uploads/             # 업로드된 PDF 저장 (자동 생성)
└── chroma_db/           # ChromaDB 벡터 저장소 (자동 생성)
```

<br/><br/>

## 📡 API

| 메서드   | 경로                  | 설명                           |
| -------- | --------------------- | ------------------------------ |
| `POST`   | `/upload`             | PDF 업로드 및 벡터 저장        |
| `POST`   | `/search`             | 자연어 질문으로 유사 청크 검색 |
| `GET`    | `/documents`          | 저장된 문서 목록 조회          |
| `DELETE` | `/documents/{doc_id}` | 문서 삭제                      |
| `GET`    | `/health`             | 서버 상태 확인                 |

<br/><br/>

## 🗺️ 로드맵

- [x] PDF 업로드 및 텍스트 추출
- [x] 청크 분할 및 임베딩 생성
- [x] ChromaDB 벡터 저장 및 유사도 검색
- [ ] LLM 연동 (GPT / Claude API) 으로 실제 답변 생성
- [ ] 멀티턴 대화 히스토리
- [ ] 출처 페이지 표시
- [ ] 배포 (Railway / Fly.io)
