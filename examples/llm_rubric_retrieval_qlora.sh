#!/usr/bin/env bash
# Rubric-retrieval with a causal LLM encoder under QLoRA.
# Paper setting: 4-bit, LoRA rank 64, lr 1e-4, 3 epochs,
# per-device batch size 4 with gradient accumulation over 8 steps.
# Requires CUDA and bitsandbytes.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
TASK=${TASK:-lp}
FIELDS=${FIELDS:-qsar}
NAME=$(basename "$MODEL")

python src/train.py \
  --task "$TASK" \
  --approach rubric-retrieval \
  --base-model "$MODEL" \
  --input-fields "$FIELDS" \
  --use-lora --use-4bit \
  --lora-rank 64 --lora-alpha 64 \
  --lr 1e-4 \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 8 \
  --bf16 \
  --output-dir "runs/${NAME}_${TASK}_${FIELDS}_rubret"
