#!/usr/bin/env bash
set -euo pipefail

API="http://localhost:8000"

echo "1) Health"
curl -s "$API/health" | python -m json.tool

echo
echo "2) List models"
curl -s "$API/models" | python -m json.tool

echo
echo "3) Chat (default model)"
curl -s "$API/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "FastAPI에서 Ollama 테스트하는 가장 간단한 방법을 한 문단으로 설명해줘."}
    ],
    "temperature": 0.2,
    "stream": false
  }' | python -m json.tool

echo
echo "4) Embeddings (default embed model)"
curl -s "$API/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["서울은 한국의 수도이다.", "FastAPI와 Ollama를 연결했다."]
  }' | python -m json.tool
