# app/rag_cli_min.py
from __future__ import annotations

import argparse
from pathlib import Path

from chroma_db import ChromaRAG


def main():
    parser = argparse.ArgumentParser(description="Minimal PDF RAG (Ollama + ChromaDB)")
    parser.add_argument("--pdf", required=True, help="PDF file path (e.g., ./pdf/sample.pdf)")
    parser.add_argument("--q", required=True, help="Question to ask")
    parser.add_argument("--chroma_dir", default="./chroma_data", help="Chroma persist directory")
    parser.add_argument("--collection", default="rag_docs", help="Chroma collection name")
    parser.add_argument("--ollama", default="http://localhost:11434", help="Ollama base url")
    parser.add_argument("--embed_model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--gen_model", default="llama3.2:3b", help="Ollama generation model")
    parser.add_argument("--top_k", type=int, default=4, help="How many chunks to retrieve")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    rag = ChromaRAG(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        ollama_base_url=args.ollama,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
    )

    # 1) PDF -> text
    pdf_bytes = pdf_path.read_bytes()
    text = rag.pdf_to_text(pdf_bytes)
    if not text.strip():
        print("PDF에서 텍스트가 거의 추출되지 않았습니다. (스캔본 PDF일 가능성) OCR이 필요할 수 있습니다.")
        return

    # 2) Ingest (chunk + embed + store)
    before = rag.count()
    added = rag.ingest_document(raw_text=text, source=str(pdf_path))
    after = rag.count()

    print(f"[INGEST] before={before}, added={added}, after={after}")

    # 3) Ask (retrieve + generate)
    out = rag.ask(args.q, top_k=args.top_k)
    print("\n===== ANSWER =====")
    print(out["answer"].strip())

    print("\n===== RETRIEVED CHUNKS (top_k) =====")
    for i, ch in enumerate(out["retrieved"], 1):
        preview = ch.strip().replace("\n", " ")
        print(f"{i}. {preview[:220]}{'...' if len(preview) > 220 else ''}")


if __name__ == "__main__":
    main()
