#!/usr/bin/env bash
set -euo pipefail

# Pull common small models from the Ollama registry.
# You can run this once after starting Ollama.

ollama pull llama3.2:3b
ollama pull nomic-embed-text
echo "Pulled default models."
