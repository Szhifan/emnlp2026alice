#!/usr/bin/env bash
# Sequence-classification baseline: sample solution + answer, fixed label head.
# Use FIELDS=qsar for the full-context (+qr) variant reported in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-jhu-clsp/mmBERT-base}
TASK=${TASK:-lp}
FIELDS=${FIELDS:-sa}
NAME=$(basename "$MODEL")

python src/train.py \
  --task "$TASK" \
  --approach seq-classification \
  --base-model "$MODEL" \
  --input-fields "$FIELDS" \
  --lr 2e-5 \
  --epochs 4 \
  --batch-size 16 \
  --output-dir "runs/${NAME}_${TASK}_${FIELDS}_seqcls"
