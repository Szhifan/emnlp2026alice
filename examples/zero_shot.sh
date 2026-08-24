#!/usr/bin/env bash
# Zero-shot prompting over the unseen-question split.
# Set OPENAI_API_KEY, or point BASE_URL at a local OpenAI-compatible server
# (vLLM, Ollama) to score with open-weight models.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-gpt-4o-mini}
TASK=${TASK:-lp}
CONFIG=${CONFIG:-full}    # full | norub | labels
SPLIT=${SPLIT:-test_uq}

python src/zero_shot_llm.py \
  --task "$TASK" \
  --split "$SPLIT" \
  --model "$MODEL" \
  --input-config "$CONFIG" \
  ${BASE_URL:+--base-url "$BASE_URL"} \
  --output-dir "runs/zeroshot_${MODEL//\//_}_${TASK}_${CONFIG}"
