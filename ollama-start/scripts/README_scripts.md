## scripts 폴더 안내

- `pull_default_models.sh` : Ollama registry에서 실습용 기본 모델을 pull
- `download_gguf.py` : Hugging Face Hub에서 GGUF 파일 다운로드 (선택)
- `create_ollama_model_from_gguf.sh` : GGUF 파일을 Ollama 커스텀 모델로 등록

macOS/Linux에서 실행 권한이 필요하면:
```bash
chmod +x scripts/*.sh examples/*.sh
```
