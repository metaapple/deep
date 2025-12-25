# chroma_db.py
from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

import requests
import chromadb
import fitz  # PyMuPDF


class ChromaRAG:
    """
    ChromaDB(영속 저장) + Ollama(임베딩/생성)을 묶은 최소 RAG 엔진.
    main.py는 라우팅만 하고, 실제 로직은 여기서 처리합니다.
    """

    def __init__(
        self,
        chroma_dir: str = "./chroma_data",
        collection_name: str = "rag_docs",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3:1b",
    ):
        # Ollama 설정
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.gen_model = gen_model

        # Chroma 설정(영속 저장)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    # -----------------------------
    # Ollama helpers
    # -----------------------------
    def embed(self, text: str) -> List[float]:
        url = f"{self.ollama_base_url}/api/embeddings"
        resp = requests.post(url, json={"model": self.embed_model, "prompt": text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return data["embedding"]

    def generate(self, prompt: str) -> str:
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.gen_model,  # gemma3:1b
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 64,  # 짧게
                # 환경에 따라 지원되는 옵션이 다를 수 있음
                # "repeat_penalty": 1.1,
            }
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")

    # -----------------------------
    # Chunking
    # -----------------------------
    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(0, end - overlap_chars)

        return chunks

    @staticmethod
    def pdf_to_text(pdf_bytes: bytes) -> str:
        """
        PDF 바이너리(bytes)를 받아서 전체 텍스트를 추출합니다.
        - 스캔본(이미지) PDF는 텍스트가 거의 안 나올 수 있음(OCR 필요)
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts).strip()


    # -----------------------------
    # Chroma operations
    # -----------------------------
    def count(self) -> int:
        return self.collection.count()

    def ingest_texts(self, texts: List[str], source: str = "manual") -> int:
        if not texts:
            return 0

        for i, t in enumerate(texts):
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[t],
                embeddings=[self.embed(t)],
                metadatas=[{"chunk": i, "source": source}],
            )
        return len(texts)

    def ingest_document(
        self,
        raw_text: str,
        source: str,
        max_chars: int = 1200,
        overlap_chars: int = 150,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta_extra = meta_extra or {}
        chunks = self.chunk_text(raw_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            return 0

        for i, ch in enumerate(chunks):
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[ch],
                embeddings=[self.embed(ch)],
                metadatas=[{"chunk": i, "source": source, **meta_extra}],
            )
        return len(chunks)

    def query_docs(self, question: str, top_k: int = 4) -> List[str]:
        n = self.count()
        if n <= 0:
            return []

        top_k = min(top_k, n)
        q_emb = self.embed(question)

        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = (res.get("documents") or [[]])[0]
        return docs


    def ask(self, question: str, top_k: int = 2) -> Dict[str, Any]:
        docs = self.query_docs(question, top_k=top_k)
        merged = "\n".join(docs)

        # 패턴형 질문은 추출 우선
        if "비밀키" in question:
            import re
            m = re.search(r"비밀키\D*([0-9]{2,})", merged)
            if m:
                return {"answer": m.group(1), "retrieved": docs, "mode": "extractive"}

        # 아니면 LLM로 생성
        context = "\n\n---\n\n".join(docs)[:3000]
        prompt = f"""(위 발췌 강제 prompt) ..."""
        answer = self.generate(prompt).strip()
        return {"answer": answer, "retrieved": docs, "mode": "generate"}
    # def ask(self, question: str, top_k: int = 2) -> Dict[str, Any]:
    #     docs = self.query_docs(question, top_k=top_k)
    #
    #     # 컨텍스트 길이를 너무 키우지 않기 위해 제한 (1B용 안전장치)
    #     context = "\n\n---\n\n".join(docs[:top_k])[:3000]
    #
    #     prompt = f"""규칙:
    # 1) 아래 CONTEXT에 있는 내용만 사용한다.
    # 2) CONTEXT에 답이 있으면 반드시 답한다. "문서에 근거가 없습니다"를 출력하면 안 된다.
    # 3) CONTEXT에 답이 없을 때만 정확히 다음 문장만 출력한다:
    # 문서에 근거가 없습니다.
    # 4) 질문이 숫자/키/포트/URL을 묻는다면 출력은 "정답만" 한 줄로 한다.
    #
    # CONTEXT:
    # {context}
    #
    # QUESTION:
    # {question}
    #
    # 정답:"""
    #
    #     answer = self.generate(prompt).strip()
    #     return {"answer": answer, "retrieved": docs}
