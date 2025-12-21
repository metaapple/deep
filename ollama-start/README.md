# Ollama + FastAPI 테스트 프로젝트 (Hugging Face GGUF 가져오기 옵션 포함)

이 저장소는 다음을 목표로 하는 **실습용 템플릿**입니다.

- 로컬에서 실행 중인 **Ollama**(LLM/Embedding)를 **FastAPI**로 호출해서 동작 여부를 빠르게 점검
- 필요하면 **Hugging Face Hub의 GGUF 모델 파일을 내려받아 Ollama에 “로컬 커스텀 모델”로 등록** 후 동일한 FastAPI 엔드포인트로 테스트
- GitHub에 바로 올릴 수 있도록 **프로젝트 구조 + 실습 예제 + 사용 매뉴얼(README)** 제공

## 1) 구성 요소

- `app/main.py` : FastAPI 엔드포인트(`/health`, `/models`, `/pull`, `/chat`, `/embed`)
- `app/ollama_client.py` : Ollama REST API 호출 래퍼
- `docker-compose.yml` : (선택) Ollama 컨테이너 + API 컨테이너 동시 실행
- `scripts/` : GGUF 다운로드/모델 생성/기본 모델 pull 스크립트
- `examples/` : curl / 파이썬 실습 예제

## 2) 빠른 시작 (로컬: Ollama는 로컬 설치)

### 2.1 Ollama 설치 및 실행
- macOS/Windows/Linux: Ollama 설치 후 실행
- 실행 확인:
  ```bash
  ollama --version
  ollama serve
  ```
  다른 터미널에서:
  ```bash
  curl http://localhost:11434/api/tags
  ```

### 2.2 기본 모델 Pull (권장)
```bash
bash scripts/pull_default_models.sh
```

### 2.3 FastAPI 실행
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### 2.4 동작 테스트
```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

또는:
```bash
bash examples/curl_examples.sh
python examples/python_client.py
```

## 3) Docker Compose로 한 번에 실행 (선택)

Ollama까지 함께 컨테이너로 띄우고 싶다면:
```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Ollama: `http://localhost:11434`

컨테이너 Ollama에 기본 모델을 미리 넣고 싶다면:
```bash
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull nomic-embed-text
```

## 4) “Hugging Face에서 가져온 모델”을 Ollama에 등록해 FastAPI에서 테스트하기

핵심 아이디어는 단순합니다.

1. Hugging Face Hub에서 **GGUF 파일**을 다운로드한다  
2. 로컬 GGUF를 `ollama create`로 **새 모델 이름**으로 등록한다  
3. FastAPI `/chat` 요청에서 `model` 값을 새 모델 이름으로 지정한다

### 4.1 GGUF 파일 다운로드

이 템플릿에는 Hugging Face Hub에서 파일을 내려받는 스크립트를 포함합니다.

```bash
# 예시 (레포/파일명은 본인이 쓰려는 GGUF에 맞게 바꾸세요)
python scripts/download_gguf.py \
  --repo <org_or_user>/<repo_name> \
  --filename <model_file>.gguf
```

- 모델이 gated/private 이면 토큰이 필요합니다:
  ```bash
  python scripts/download_gguf.py --repo ... --filename ... --token $HF_TOKEN
  ```

다운로드 결과는 `./models/` 아래에 저장됩니다.

### 4.2 Ollama 커스텀 모델 생성 (GGUF -> Ollama 모델)

```bash
# <new_model_name>을 원하는 이름으로
bash scripts/create_ollama_model_from_gguf.sh <new_model_name> ./models/<model_file>.gguf
```

생성 확인:
```bash
ollama list
```

### 4.3 FastAPI에서 커스텀 모델로 채팅 테스트

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<new_model_name>",
    "messages": [
      {"role":"system","content":"You are helpful."},
      {"role":"user","content":"GGUF로 만든 커스텀 모델이 FastAPI에서 잘 호출되는지 테스트 문장을 만들어줘."}
    ],
    "temperature": 0.2,
    "stream": false
  }' | python -m json.tool
```

## 5) 실습 시나리오 (권장 순서)

1. `/health`로 **Ollama 연결 확인**
2. `/models`로 **모델 목록 확인**
3. `/chat`으로 **기본 모델** 응답 확인
4. `/embed`로 **임베딩 벡터 차원/개수** 확인
5. (선택) Hugging Face GGUF → `ollama create` → `/chat`에서 `model` 변경하여 응답 비교

## 6) 자주 발생하는 문제와 해결

### 6.1 `/health`에서 `ollama_reachable=false`
- Ollama가 실행 중인지 확인: `ollama serve`
- 포트 충돌 확인: `lsof -i :11434` (macOS/Linux)
- Docker Compose라면 API 컨테이너 환경변수 `OLLAMA_BASE_URL=http://ollama:11434` 확인

### 6.2 `/chat`에서 503 혹은 모델 not found
- Ollama에 해당 모델이 존재하는지 확인: `ollama list`
- 없다면:
  - 레지스트리 모델: `ollama pull <model>`
  - GGUF 커스텀: `ollama create <name> -f Modelfile` (스크립트 사용 권장)

### 6.3 GGUF 파일이 너무 커서 느림/메모리 부족
- 더 낮은 quantization(Q4 등) GGUF 선택
- `num_ctx`/옵션을 낮춰 테스트
- 모델 자체가 너무 큰 경우 더 작은 모델로 실습

## 7) API 요약

- `GET /health`
- `GET /models`
- `POST /pull` : Ollama registry 모델 pull (GGUF는 별도)
- `POST /chat`
- `POST /embed`

## 8) 라이선스

이 템플릿 코드는 MIT로 사용해도 무방합니다.  
단, Hugging Face에서 내려받는 모델 파일/가중치의 라이선스는 각 모델 저장소의 정책을 따르세요.
