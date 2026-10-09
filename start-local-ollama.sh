#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if curl --silent --fail --max-time 2 --noproxy '*' http://127.0.0.1:11434/api/tags >/dev/null; then
    echo 'Ollama already listens on 127.0.0.1:11434; reusing it. No services changed.'
    exit 0
fi
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NO_CLOUD=1
export OLLAMA_MODELS="$PWD/.local-data/ollama-models"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
# Vulkan acceleration was verified on this workstation's Radeon 780M.
# Override with OLLAMA_VULKAN=0 OLLAMA_IGPU_ENABLE=0 for CPU mode.
export OLLAMA_IGPU_ENABLE="${OLLAMA_IGPU_ENABLE:-1}"
export OLLAMA_VULKAN="${OLLAMA_VULKAN:-1}"
if command -v ollama >/dev/null; then
    exec ollama serve
fi
exec .tools/ollama/bin/ollama serve
