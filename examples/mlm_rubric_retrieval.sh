#!/usr/bin/env bash
# Rubric-retrieval with a masked LM encoder.
# Paper setting: lr 2e-5, 4 epochs, batch size 16.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-jhu-clsp/mmBERT-base}
TASK=${TASK:-lp}          # lp | ke | sk
FIELDS=${FIELDS:-ar}      # ar (base) or qsar (full context)
NAME=$(basename "$MODEL")

python src/train.py \
  --task "$TASK" \
  --approach rubric-retrieval \
  --base-model "$MODEL" \
  --input-fields "$FIELDS" \
  --lr 2e-5 \
  --epochs 4 \
  --batch-size 16 \
  --output-dir "runs/${NAME}_${TASK}_${FIELDS}_rubret"
