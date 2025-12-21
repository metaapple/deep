# main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download

from chroma_db import ChromaRAG
from schemas import IngestTextsRequest, IngestHFRequest, IngestPDFRequest, AskRequest, AskResponse
from fastapi import UploadFile, File


# RAG 엔진(전역 1개)
rag = ChromaRAG(
    chroma_dir="./chroma_data",
    collection_name="rag_docs",
    ollama_base_url="http://localhost:11434",
    embed_model="nomic-embed-text",
    gen_model="llama3.2:3b",
)

app = FastAPI(title="Ollama + Chroma RAG (3 files)")

@app.get("/")
def root():
    return {"message": "welcome"}

@app.get("/health")
def health():
    return {"ok": True, "docs": rag.count()}

@app.post("/ingest_texts")
def ingest_texts(req: IngestTextsRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")

    added = rag.ingest_texts(req.texts, source=req.source)
    return {"docs_added": added, "total_docs": rag.count()}

@app.post("/ingest_hf")
def ingest_hf(req: IngestHFRequest):
    # HF에서 파일 다운로드
    try:
        local_path = hf_hub_download(repo_id=req.repo_id, filename=req.filename, revision=req.revision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF download failed: {e}")

    # 파일 읽기
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {e}")

    # 청킹 후 저장
    chunks_added = rag.ingest_document(
        raw_text=text,
        source="huggingface",
        max_chars=req.max_chars,
        overlap_chars=req.overlap_chars,
        meta_extra={"repo_id": req.repo_id, "filename": req.filename, "revision": req.revision or ""},
    )

    return {"chunks_added": chunks_added, "total_docs": rag.count()}


@app.post("/ingest_pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    max_chars: int = 1200,
    overlap_chars: int = 150,
    source: str = "pdf",
):
    """
    PDF 파일을 업로드 받아 텍스트 추출 → 청킹 → Chroma 저장
    - 파라미터는 query string 형태로도 받을 수 있게 최소로 구성
    - 예: /ingest_pdf?max_chars=1200&overlap_chars=150&source=pdf
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # PDF → 텍스트 추출
    text = rag.pdf_to_text(pdf_bytes)
    if not text.strip():
        # 스캔 PDF(이미지)면 텍스트가 없을 수 있음
        raise HTTPException(
            status_code=400,
            detail="No extractable text found. If it's a scanned PDF, OCR is needed."
        )

    # 청킹 후 저장
    chunks_added = rag.ingest_document(
        raw_text=text,
        source=source,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        meta_extra={"filename": file.filename},
    )

    return {"chunks_added": chunks_added, "total_docs": rag.count(), "filename": file.filename}


from fastapi import HTTPException
from schemas import AskRequest  # AskResponse는 굳이 안 써도 됨

@app.post("/ask")
def ask(req: AskRequest):
    # 문서가 하나도 없으면 질문해도 의미가 없으니 400 처리
    if rag.count() == 0:
        raise HTTPException(status_code=400, detail="No documents. Ingest first.")

    # RAG 실행 (검색 -> 프롬프트 -> 생성)
    out = rag.ask(req.question, top_k=req.top_k)

    # dict로 그대로 반환하면 FastAPI가 JSON으로 바꿔서 응답함
    # out 구조: {"answer": "...", "retrieved": ["...", "..."]}
    return out

@app.post("/search")
def search(req: AskRequest):
    if rag.count() == 0:
        raise HTTPException(status_code=400, detail="No documents. Ingest first.")
    docs = rag.query_docs(req.question, top_k=req.top_k)
    return {"retrieved": docs}

