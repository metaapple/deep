import ollama
import asyncio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


class HealthResponse(BaseModel):
    status : str
    ollama_status : str
    message: str

app = FastAPI()

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates 설정
templates = Jinja2Templates(directory="templates")


# 사용할 모델 이름 (미리 ollama pull <model>로 다운로드 필요, 예: ollama pull llama3.2)
# MODEL = "llama3.2"
MODEL = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

@app.get("/")
def read_root(request : Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/health")
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
        ollama_status = "타임아웃"
        message = "Ollama 타임 아웃"
    except Exception as e:
        ollama_status = "error"
        message = f"Error checking Ollama: {str(e)}"

    return HealthResponse(
        status="ok",
        ollama_status=ollama_status,
        message=message
    )
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
@app.get("/chat")
async def generate(word : str, request : Request):
    try:
        response = await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=word,
            options={"temperature": 1},
            keep_alive=-1  # 필요 시 후속 요청에서도 유지
        )
        return templates.TemplateResponse("chat.html",
                                      context={"request": request,
                                               "result" : response["response"]
                                               })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 스트리밍 엔드포인트 (실시간 토큰 반환, 더 빠른 체감)
from fastapi.responses import StreamingResponse

@app.get("/stream")
async def stream(word : str):
    return StreamingResponse(stream_generate(word), media_type="text/event-stream")

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

@app.get("/ollama-test")
def ollama_test(request : Request):
    return templates.TemplateResponse("ollama-test.html", context={"request": request})


# 추가 엔드포인트
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 간단히 세션 구분


# 메모리 기반 간단한 대화 히스토리 저장 (프로덕션에서는 Redis 등 사용)
chat_histories = {}


@app.post("/chat")
async def chat(request: ChatRequest):
    history = chat_histories.get(request.session_id, [])
    history.append({"role": "user", "content": request.message})

    response = await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )

    ai_message = response["message"]["content"]
    history.append({"role": "assistant", "content": ai_message})
    chat_histories[request.session_id] = history[-10:]  # 최근 10턴 유지

    return {"response": ai_message}

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 200

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    prompt = f"다음 텍스트를 한국어로 {request.max_length}자 이내로 요약해 주세요:\n\n{request.text}"
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"summary": response["response"].strip()}

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    prompt = f"다음 영어 문장을 자연스러운 한국어로 번역해 주세요. 번역만 출력하세요:\n\n{request.text}"
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"translation": response["response"].strip()}


class SentimentRequest(BaseModel):
    text: str


@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    prompt = f"""
    다음 문장의 감정을 분석해 주세요. 답변은 반드시 다음 중 하나만 출력하세요: 긍정, 부정, 중립

    문장: {request.text}
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    sentiment = response["response"].strip()
    return {"sentiment": sentiment}

class BrainstormRequest(BaseModel):
    topic: str
    count: int = 5

@app.post("/brainstorm")
async def brainstorm(request: BrainstormRequest):
    prompt = f"""
    주제 '{request.topic}'에 대해 창의적이고 실현 가능한 아이디어를 {request.count}개 제안해 주세요.
    각 아이디어는 번호를 붙이고 한 문장으로 간단히 설명하세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"ideas": response["response"].strip()}


# 6. 시 쓰기 도우미
class PoemRequest(BaseModel):
    topic: str
    style: str = "현대시"  # 예: 현대시, 전통 시조, 자유시 등


@app.post("/poem")
async def write_poem(request: PoemRequest):
    prompt = f"""
    다음 주제로 한국어로 아름다운 시를 한 편 지어주세요.
    스타일은 '{request.style}'로 해주세요. 감성적이고 운율이 살아있게 해주세요.

    주제: {request.topic}

    제목도 함께 붙여주세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"poem": response["response"].strip()}


# 7. 레시피 생성기
class RecipeRequest(BaseModel):
    ingredients: str  # 예: "계란, 토마토, 양파, 치즈"
    servings: int = 2
    difficulty: str = "쉬움"  # 쉬움, 보통, 어려움


@app.post("/recipe")
async def generate_recipe(request: RecipeRequest):
    prompt = f"""
    다음 재료를 사용해서 {request.servings}인분 요리를 만들어 주세요.
    난이도는 '{request.difficulty}' 수준으로, 단계별로 자세히 설명해 주세요.

    재료: {request.ingredients}

    요리 이름도 창의적으로 지어주고, 필요한 추가 재료(조미료 등)는 최소한으로 제안해 주세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"recipe": response["response"].strip()}


# 8. 이름 지어주기 도우미
class NameRequest(BaseModel):
    category: str = "아기"  # 아기, 강아지, 고양이, 프로젝트, 카페, 브랜드 등
    gender: str = "중성"  # 남자, 여자, 중성
    count: int = 5
    vibe: str = ""  # 예: 귀여운, 세련된, 강한, 따뜻한 등 (선택)


@app.post("/names")
async def suggest_names(request: NameRequest):
    vibe_text = f", {request.vibe} 느낌" if request.vibe else ""
    prompt = f"""
    {request.category} 이름을 {request.gender} 성별에 맞춰 {request.count}개 제안해 주세요.
    한국어 이름으로 자연스럽고 예쁜 이름 위주로 해주세요{vibe_text}.
    각 이름 옆에 간단한 의미나 이유도 함께 설명해 주세요.

    예시 형식:
    1. 하율 - 하늘처럼 맑고 넓은 마음을 가진 아이
    2. ...
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"names": response["response"].strip()}