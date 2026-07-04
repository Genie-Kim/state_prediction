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
TRAJECTORY_JSONL="${TRAJECTORY_JSONL:?set TRAJECTORY_JSONL=/path/to/trajectory.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/nsr_sft}"
PREPARED_DATASET="${PREPARED_DATASET:-$ROOT_DIR/data/nsr_sft_train.jsonl}"
SPECIAL_TOKENS_FILE="${SPECIAL_TOKENS_FILE:-$OUTPUT_DIR/nsr_special_tokens.txt}"

$PYTHON_BIN -m nsr.qwen_guard "$MODEL"
$PYTHON_BIN training/sft/write_special_tokens.py \
  --nsr-config "$NSR_CONFIG" \
  --output "$SPECIAL_TOKENS_FILE"

if [[ "${PREPARE_DATASET:-1}" == "1" ]]; then
  $PYTHON_BIN training/sft/prepare_dataset.py \
    --input "$TRAJECTORY_JSONL" \
    --output "$PREPARED_DATASET" \
    --nsr-config "$NSR_CONFIG"
fi

$PYTHON_BIN -m swift.cli.sft \
  --external_plugins "$ROOT_DIR/training/sft/plugin.py" \
  --model "$MODEL" \
  --dataset "$PREPARED_DATASET" \
  --loss_type nsr_sft \
  --remove_unused_columns false \
  --new_special_tokens "$SPECIAL_TOKENS_FILE" \
  --tuner_type lora \
  --target_modules all-linear \
  --modules_to_save all-embedding lm_head \
  --lora_rank "${LORA_RANK:-128}" \
  --lora_alpha "${LORA_ALPHA:-256}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --eval_steps "${EVAL_STEPS:-500}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --max_length "${MAX_LENGTH:-4096}" \
  --callbacks nsr_save \
  --output_dir "$OUTPUT_DIR" \
  "$@"
