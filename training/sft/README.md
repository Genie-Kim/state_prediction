# NSR SFT

This folder trains the one-shot special-token retrieval model with ms-swift.

Flow:

1. Precompute DINOv3 CLS latents with `training/precompute_latents.py`.
2. Convert trajectory JSONL to ms-swift JSONL with `prepare_dataset.py`.
3. Run `run_sft.sh`, which registers `training/sft/plugin.py`.

The plugin adds `nsr_sft` loss:

`lambda_lm * CE(narration, excluding <emb>) + lambda_ret * InfoNCE(q_proj(hidden@<emb>), DINOv3_CLS(target))`

Set `num_emb_tokens=1` for the main reusable `<|emb|>` mode. Set `num_emb_tokens` from 2 to 10 to use numbered `<|emb_1|>` ... `<|emb_N|>` for ablations.
