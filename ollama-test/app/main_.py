from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import asyncio

app = FastAPI()

# 사용할 모델 이름 (미리 ollama pull <model>로 다운로드 필요, 예: ollama pull llama3.2)
MODEL = "llama3.2"

class PromptRequest(BaseModel):
    prompt: str

# 앱 시작 시 모델 미리 로드 (preload)
@app.on_event("startup")
async def preload_model():
    try:
        # 빈 프롬프트로 모델 로드 + 영구 유지
        await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=" ",  # 빈 프롬프트 (또는 "preload" 같은 더미 텍스트)
            keep_alive=-1  # -1: 영구적으로 메모리에 유지
        )
        print(f"{MODEL} 모델이 미리 로드되었습니다. (메모리에 영구 유지)")
    except Exception as e:
        print(f"모델 preload 실패: {e}")

# 일반 generate 엔드포인트 (스트리밍 없이 전체 응답)
@app.post("/chat")
async def generate(request: PromptRequest):
    try:
        response = await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=request.prompt,
            options={"temperature": 0.7},
            keep_alive=-1  # 필요 시 후속 요청에서도 유지
        )
        return {"response": response["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 스트리밍 엔드포인트 (실시간 토큰 반환, 더 빠른 체감)
from fastapi.responses import StreamingResponse

async def stream_generate(prompt: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        keep_alive=-1
    )
    async for part in stream:
        # "이 값을 내보내고, 여기서 잠깐 멈춰. 다음에 다시 불러주면 이어서 할게!"
        # ollama로 부터 받은 조각마다 보내..
        yield part["response"]

        # 한꺼번에 보내고 싶으면
        # async def stream_generate_bad(prompt: str):
        #     full_text = ""
        #     stream = await ollama.AsyncClient().generate(...)
        #     async for part in stream:
        #         full_text += part["response"]
        #     return full_text  # ← 전체 다 모아서 한 번에 리턴

@app.post("/stream")
async def stream(request: PromptRequest):
    return StreamingResponse(stream_generate(request.prompt), media_type="text/event-stream")