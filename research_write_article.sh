#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" || exit 1
source venv/bin/activate

if ! curl -sf http://127.0.0.1:11434 > /dev/null; then
  echo "Ollama isn't running, starting it..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  for _ in {1..30}; do
    curl -sf http://127.0.0.1:11434 > /dev/null && break
    sleep 1
  done
  curl -sf http://127.0.0.1:11434 > /dev/null || {
    echo "Ollama failed to start within 30s. See /tmp/ollama.log"
    exit 1
  }
fi

echo "Ollama is up."

REQUIRED_MODELS=("llama3.1:8b")
INSTALLED=$(ollama list | awk 'NR>1 {print $1}')
for model in "${REQUIRED_MODELS[@]}"; do
  if ! echo "$INSTALLED" | grep -qx "$model"; then
    echo "Model $model not found, pulling..."
    ollama pull "$model"
  fi
done

python research_write_article.py "$@"
