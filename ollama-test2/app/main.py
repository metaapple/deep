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
from ollama_client import stream_generate
from redis.asyncio import Redis  # redis-py의 async 클라이언트
import json

# from fastapi.middleware.cors import CORSMiddleware
# from transformers import pipeline




class HealthResponse(BaseModel):
    status : str
    ollama_status : str
    message: str

app = FastAPI()

# Redis 설정 (로컬 개발 기준, 프로덕션에서는 환경변수로 관리)
REDIS_URL = "redis://localhost:6379"
# 앱 상태에 Redis 클라이언트 저장
app.state.redis = None

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates 설정
templates = Jinja2Templates(directory="templates")

# CORS 설정 (클라이언트 도메인 허용, 개발 시 "*"로 테스트)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한, 예: ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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

        app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
        print("Redis 연결 성공")

    except Exception as e:
        print(f"모델 preload 실패 또는 Redis연결 실패 : {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if app.state.redis:
        await app.state.redis.close()

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

## 파라메터 전달용 class를 만들자.
## 파라메터 이름 똑같은거 자동으로 변수에 들어감.
## 다른 옵션값들 설정 가능
## BaseModel이라는 클래스를 상속받아서 만들어야 자동으로
## 이런 처리들을 해줌.

class SummarizeRequest(BaseModel):
    ## BaseModel(변수+함수) + 내가 추가한 변수
    text : str
    max_length : int = 200


@app.post("/summarize")
async def summarize(request : SummarizeRequest):
    # http://localhost:11434/api/generate, json=payload
    # post방식으로 http요청을 해줌.
    prompt = f"{request.text}를 {request.max_length}자로 요약해주세요."
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------")
    print(response) #dict
    return {'summary' : response["response"].strip()}

class NameRequest(BaseModel):
    # axios.post로 전달될 때 키와 이름이 같아야한다.
    # {category : "아기", gender : "여성", ....}
    category : str = "카페"
    gender : str = "중성"
    count : int = 3
    vibe : str = "따뜻한"

@app.post("/names")
async def names(request : NameRequest):
    prompt = f"""
                {request.category}이름을 
                {request.gender}, {request.vibe}느낌으로 
                {request.count}개만 추천해줘.
            
            결과 화면은 다음과 같이 만들어줘.
            
            1. 이름 - 간단설명
            2. 이름 - 간단설명
            3. 이름 - 간단설명
            """
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )

    print("-----------------")
    print(response)  # dict
    return {'names': response["response"].strip()}


# # 요약 파이프라인 (한 번만 로드)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# def extract_text_from_url(url: str) -> str:
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#
#     # 간단한 추출 (사이트별로 다름, 더 정확하게 하려면 readability 사용)
#     for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
#         script.decompose()
#
#     text = soup.get_text(separator='\n')
#     lines = (line.strip() for line in text.splitlines())
#     chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#     text = '\n'.join(chunk for chunk in chunks if chunk)
#     return text

# class TextRequest(BaseModel):
#     text: str

# @app.post("/summarize-text")
# async def summarize_text(request: TextRequest):
#     # 예: transformers로 요약
#     summary = summarizer(request.text[:1000], max_length=300, min_length=50, do_sample=False)[0]['summary_text']
#     return {"summary": summary}

# 추가 엔드포인트
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 간단히 세션 구분

# 메모리 기반 간단한 대화 히스토리 저장 (프로덕션에서는 Redis 등 사용)
chat_histories = {}

########### redis연결전
# @app.post("/chat")
async def chat(request: ChatRequest):
    history = chat_histories.get(request.session_id, [])
    history.append({"role": "user", "content": request.message + ", 200글자 이내로 핵심만 답을 줘."})
    # 나는 user, ai는 assistant

    response = await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )

    print("-----------------")
    print(response)
    # message = Message(role='assistant',
    #                   content='싱가포르는 매력적인 도시로, 다양한 경험을 제공하는 매혹적인 나라입니다. 싱가포르 여행에 대한 유용한 정보들을 정리해 드릴게요.\n\n**1. 여행 준비**\n\n*   **비자:** 한국인은 90일까지 무비자 체류 가능합니다
    ai_message = response["message"]["content"]
    history.append({"role": "assistant", "content": ai_message})
    chat_histories[request.session_id] = history[-10:]  # 최근 10턴 유지

    print("chat_histories>> ", chat_histories)
    # chat_histories >> {'default': [{'role': 'user', 'content': '싱가폴 여행정보, 200글자 이내로 핵심만 답을 줘.'},
                        # {'role': 'assistant', 'content': '싱가포르는 다채로운 문화와 현대적인 도시, 그리고 맛있는 음식으로 유명합니다. \n\n*   **교통:** 대중교통 시스템이 매우 잘 갖춰져 있어 편리하게 이동할 수 있습니다.\n*   **관광 명소:** 마리나 베이 푹, 센토사 섬, 칠리 궁전 등 다양한 명소가 있습니다.\n*   **특징:** 24시간 영업하는 식료품점, 다양한 길거리 음식, 럭셔리한 쇼핑 등 독특한 경험을 할 수 있습니다.\n\n**여행 준비:** 미리 항공권과 숙소를 예약하고, 환전은 한국 원화로 하는 것이 좋습니다.'}]}
    return {"response": ai_message}


# 기존 /chat 엔드포인트 수정
########### redis연결후
@app.post("/chat")
async def chat(request: ChatRequest):
    redis = app.state.redis
    if not redis:
        raise HTTPException(status_code=500, detail="Redis 연결 안됨")

    session_key = f"chat_history:{request.session_id}"

    # Redis에서 기존 히스토리 불러오기 (List로 저장, 왼쪽이 오래된 메시지)
    history_json = await redis.lrange(session_key, 0, -1) #리스트의 처음부터 끝까지 다 가지고 오기
    print("history_json>> ", history_json)
    history = [json.loads(msg) for msg in history_json]
    print("history>> ", history)

    # 사용자 메시지 추가
    user_message = request.message + ", 200글자 이내로 핵심만 답을 줘."
    history.append({"role": "user", "content": user_message})

    # Ollama 호출
    response = await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )

    ai_message = response["message"]["content"]
    history.append({"role": "assistant", "content": ai_message})

    # Redis에 다시 저장 (최근 10턴 = 20 메시지만 유지)
    # 먼저 기존 리스트 지우고 새로 push
    await redis.delete(session_key)
    # 최근 20개만 저장 (역순으로 push해서 lrange로 오래된->최신 순서 유지)
    recent_history = history[-20:]
    if recent_history:
        await redis.rpush(session_key, *[json.dumps(msg) for msg in recent_history])

    # 선택적으로 TTL 설정 (예: 7일 후 자동 만료)
    await redis.expire(session_key, 60 * 60 * 24 * 7)

    return {"response": ai_message}

@app.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    redis = app.state.redis
    if not redis:
        raise HTTPException(status_code=500, detail="Redis 연결 안됨")

    session_key = f"chat_history:{session_id}"
    history_json = await redis.lrange(session_key, 0, -1)
    history = [json.loads(msg) for msg in history_json]

    # role에 따라 구분해서 프론트에서 쉽게 표시할 수 있게
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append({"type": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            messages.append({"type": "ai", "content": msg["content"]})

    return {"history": messages}


class TranslateRequest(BaseModel):
    text:str
@app.post('/translate')
async def translate(request: TranslateRequest):
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"{request.text}를 자연스러운 한국어로 번역해주세요"
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}

class NamesRequest(BaseModel):
    #axios.post로 전달될 때 키와 이름이 같아야 한다.
    category:str='카페'
    gender:str="중성"
    vibe:str='귀여운'
    count:int=3
    
@app.post('/create_name')
async def translate(request: NamesRequest):
    print('request',request.category)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"""{request.category}의 이름을 
                {request.gender}느낌으로 
                {request.count}개만 추천해주세요.
            """
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}

class PoemsRequest(BaseModel):
    topic:str
    style:str
    
@app.post('/poem')
async def translate(request: PoemsRequest):
    print('request',request.topic,request.style)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"""{request.topic}을 주제로 
                {request.style}스타일로 시를 지어 주세요.
            """
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}
