
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from ollama_client import ollama_client

app = FastAPI()

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates 설정
templates = Jinja2Templates(directory="templates")


# Ollama 기본 설정
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 기본 포트


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """FastAPI와 Ollama의 health 상태를 확인하는 엔드포인트"""
    try:
        # Ollama health check
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)

        if response.status_code == 200:
            ollama_status = "healthy"
            message = "fastapi & ollama 제대로 동작중"
        else:
            ollama_status = "unhealthy"
            message = f"Ollama returned status code: {response.status_code}"

    except httpx.ConnectError:
        ollama_status = "연결불가"
        message = "Ollama 연결할 수 없음."
    except httpx.TimeoutException:
        ollama_status = "타임 아웃"
        message = "Ollama 타임 아웃"
    except Exception as e:
        ollama_status = "error"
        message = f"Error checking Ollama: {str(e)}"

    return HealthResponse(
        status="ok",
        ollama_status=ollama_status,
        message=message
    )

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
def chat(request: Request, word:str):
    result = ollama_client(word)
    ### ollama 결과
    print(result.get('success'))
    return templates.TemplateResponse("chat.html", {"request": request, "result": result.get('answer')})

@app.get("/chat2")
def chat2(request: Request, word:str):
    result = ollama_client(word)
    ### ollama 결과
    print(result.get('success'))
    print(result.get('answer'))
    return  {"result": result.get('answer')}

if __name__ == '__main__':
    uvicorn.run(app)