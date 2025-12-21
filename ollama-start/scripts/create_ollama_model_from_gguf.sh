#!/usr/bin/env bash
set -euo pipefail

# Create an Ollama model from a local GGUF file.
#
# Prereqs:
# - Ollama installed and running (ollama serve)
# - You have a GGUF file under ./models
#
# Usage:
#   bash scripts/create_ollama_model_from_gguf.sh my-hf-model ./models/model.Q4_K_M.gguf
#
# After create:
#   ollama list
#   curl http://localhost:11434/api/generate -d '{"model":"my-hf-model","prompt":"Hello","stream":false}'

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <new_model_name> <path_to_gguf>"
  exit 1
fi

MODEL_NAME="$1"
GGUF_PATH="$2"

if [[ ! -f "$GGUF_PATH" ]]; then
  echo "GGUF not found: $GGUF_PATH"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cat > "$TMP_DIR/Modelfile" <<EOF
FROM $GGUF_PATH

# Optional inference defaults:
PARAMETER temperature 0.2
PARAMETER num_ctx 4096

SYSTEM """You are a helpful assistant. Answer in Korean unless the user asks otherwise."""
EOF

echo "Creating Ollama model '$MODEL_NAME' from GGUF: $GGUF_PATH"
ollama create "$MODEL_NAME" -f "$TMP_DIR/Modelfile"
echo "Done. Try: ollama run $MODEL_NAME"
