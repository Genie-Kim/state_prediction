#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="uv run python"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/ms-swift:${PYTHONPATH:-}"
export NSR_CONFIG="${NSR_CONFIG:-$ROOT_DIR/configs/nsr_qwen3vl_reusable_emb.json}"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DATASET="${DATASET:?set DATASET=/path/to/ms-swift-nsr.jsonl}"

$PYTHON_BIN -m nsr.qwen_guard "$MODEL"

$PYTHON_BIN -m swift.cli.rlhf \
  --rlhf_type grpo \
  --external_plugins "$ROOT_DIR/training/grpo/plugin.py" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --reward_funcs nsr_emb_format \
  --remove_unused_columns false \
  --tuner_type lora \
  --target_modules all-linear \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
  --output_dir "${OUTPUT_DIR:-$ROOT_DIR/output/nsr_grpo}" \
  "$@"
