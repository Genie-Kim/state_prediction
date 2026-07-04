# State Prediction / Next-State Retrieval

This repo implements the proposal in `proposal_next_state_retrieval.md`:
Qwen3.5/Qwen3-VL reads the final process image and process steps, places
special `<emb>` tokens in a deterministic assistant response, and trains the
hidden state at each `<emb>` to retrieve the matching intermediate-state DINOv3
CLS latent.

The main mode is a single reusable token:

- prompt marker: `<|slot|>`
- response query token: `<|emb|>`

For ablations, set `num_emb_tokens` to 2..10 in the NSR config to use
`<|emb_1|>` ... `<|emb_N|>`.

## Folders

- `nsr/`: shared schema, prompt builders, DINO latent store, projection head,
  and InfoNCE helpers.
- `training/sft/`: ms-swift external plugin and SFT launch script for the
  retrieval loss.
- `training/grpo/`: one-shot format reward skeleton for GRPO. Retrieval loss is
  intentionally handled by SFT because it needs hidden states.
- `inference/`: one-shot retrieval inference; no generation loop or multi-turn
  rollout.
- `evaluation/`: Recall@K evaluation over trajectory JSONL.

## Minimal Flow

```bash
export PYTHONPATH="$PWD:$PWD/ms-swift:${PYTHONPATH:-}"

.venv/bin/python training/precompute_latents.py \
  --data data/trajectories.jsonl \
  --output latents/nsr_train \
  --dino-model facebook/dinov3-vitb16-pretrain-lvd1689m

NSR_CONFIG=configs/nsr_qwen3vl_reusable_emb.json \
TRAJECTORY_JSONL=data/trajectories.jsonl \
MODEL=Qwen/Qwen3.5-4B \
training/sft/run_sft.sh

.venv/bin/python inference/retrieve.py \
  --model Qwen/Qwen3.5-4B \
  --adapter output/nsr_sft/checkpoint-500 \
  --data data/trajectories.jsonl \
  --top-k 10
```

Models are guarded for Qwen3.5/Qwen3-VL family names and <=30B-scale names in
the provided scripts.
