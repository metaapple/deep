import requests

API = "http://localhost:8000"

print("[health]")
print(requests.get(f"{API}/health").json())

print("\n[chat]")
payload = {
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Ollama와 FastAPI 연결이 잘 되었는지 확인하는 체크리스트를 5개만."},
    ],
    "temperature": 0.2,
    "stream": False,
}
r = requests.post(f"{API}/chat", json=payload)
r.raise_for_status()
print(r.json()["response"])

print("\n[embed]")
payload = {"input": ["문장 임베딩 테스트", "서울에서 개발 중입니다."]}
r = requests.post(f"{API}/embed", json=payload)
r.raise_for_status()
embs = r.json()["embeddings"]
print("embeddings:", len(embs), "vectors;", "dim=", len(embs[0]) if embs else None)
