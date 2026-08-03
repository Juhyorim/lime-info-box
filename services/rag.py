from clients import openai_client
from config import current_model


def call_llm(system_prompt: str, user_prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=current_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=0.3
    )
    return response.choices[0].message.content


# ── RAG 프롬프트 ──────────────────────────────────────────
def build_rag_prompt(query: str, chunks: list, history: list) -> tuple[str, str]:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[출처 {i+1}: {chunk['filename']} | 청크 #{chunk['chunk_index']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)

    history_text = ""
    if history:
        history_lines = []
        for turn in history[-6:]:
            history_lines.append(f"사용자: {turn['user']}")
            history_lines.append(f"AI: {turn['assistant']}")
        history_text = "\n".join(history_lines)

    system_prompt = """당신은 개인 지식베이스 AI 비서입니다.
반드시 아래 규칙을 따르세요:
1. 반드시 제공된 컨텍스트 문서만을 근거로 답변하세요.
2. 컨텍스트에 없는 내용은 절대 추측하거나 지어내지 마세요.
3. 컨텍스트에 답이 없으면 "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 말하세요.
4. 반드시 한국어로 답변하세요.
5. 답변 마지막에 참고한 출처를 명시하세요."""

    user_prompt = f"""[이전 대화]
{history_text if history_text else "없음"}

[참고 문서]
{context}

[질문]
{query}"""

    return system_prompt, user_prompt


# ── 대화 히스토리 ─────────────────────────────────────────
conversation_store: dict[str, list] = {}
