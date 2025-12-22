
from fastapi import HTTPException
import requests


# Ollama 기본 설정
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 기본 포트
# DEFAULT_MODEL = "llama3.2"
DEFAULT_MODEL = "gemma2:2b"
# 명령프롬프트 --> ollama pull gemma2:2b

"""
word 파라미터를 받아서 Ollama에 전송하고 응답을 JSON으로 반환

Args:
    word: 사용자가 입력한 질문/메시지

Returns:
    JSON 응답 (질문, 답변, 메타데이터 포함)
"""

def ollama_client(word : str):
    if not word or word.strip() == "":
        raise HTTPException(status_code=400, detail="word 파라미터가 비어있습니다.")

    try:
        # Ollama API에 전송할 페이로드
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": word.strip(),
            "stream": False,  # 스트리밍 비활성화로 전체 응답 받기
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }

        response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=60.0  # 응답 대기 시간
            )
        if response.status_code == 200:
            ollama_response = response.json()

        # JSON 응답 구성
        result = {
            "success": True,
            "question": word,
            "answer": ollama_response.get("response", ""),
            "model": ollama_response.get("model", DEFAULT_MODEL),
            "metadata": {
                "total_duration": ollama_response.get("total_duration"),
                "load_duration": ollama_response.get("load_duration"),
                "prompt_eval_count": ollama_response.get("prompt_eval_count"),
                "eval_count": ollama_response.get("eval_count"),
                "eval_duration": ollama_response.get("eval_duration")
            }
        }
        return result
    except:
        return {"success": False}


