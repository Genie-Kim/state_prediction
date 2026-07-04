"""One-shot NSR retrieval inference.

This script does not generate a multi-turn response. It builds the deterministic
assistant response containing <emb> tokens, runs one forward pass, projects the
<emb> hidden states, and retrieves nearest DINOv3 gallery entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ms-swift"))

import numpy as np
import torch

from nsr.config import NSRConfig
from nsr.constants import special_token_names
from nsr.latent_store import LatentStore
from nsr.losses import extract_emb_hidden_states
from nsr.projection import load_q_proj
from nsr.retrieval import search_topk
from nsr.schema import TrajectorySample, build_messages, iter_jsonl
from nsr.swift_utils import (
    emb_token_ids,
    model_device,
    model_hidden_size,
    move_to_device,
    output_last_hidden_state,
)


def load_runtime(
    model_id: str,
    cfg: NSRConfig,
    adapter: str | None = None,
    q_proj_dir: str | None = None,
    torch_dtype: str | None = None,
    attn_impl: str | None = None,
) -> dict[str, Any]:
    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template

    kwargs: dict[str, Any] = {
        "model": model_id,
        "infer_backend": "transformers",
        "new_special_tokens": special_token_names(cfg.num_emb_tokens),
    }
    if adapter:
        kwargs["adapters"] = [adapter]
    if torch_dtype:
        kwargs["torch_dtype"] = torch_dtype
    if attn_impl:
        kwargs["attn_impl"] = attn_impl
    infer_args = InferArguments(**kwargs)
    model, template = prepare_model_template(infer_args)
    template.set_mode("train")
    model.eval()

    q_proj_root = q_proj_dir or adapter
    if not q_proj_root:
        raise ValueError("--q-proj-dir is required when --adapter is not set")
    q_proj = load_q_proj(cfg, model_hidden_size(model), q_proj_root, device=model_device(model))
    q_proj.eval()
    store = LatentStore(cfg.latent_store, mmap=True)
    return {"model": model, "template": template, "q_proj": q_proj, "store": store}


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    skip = {"labels", "loss_scale", "channel"}
    return {k: v for k, v in batch.items() if k not in skip}


@torch.inference_mode()
def retrieve_sample(
    sample: TrajectorySample,
    cfg: NSRConfig,
    runtime: dict[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    model = runtime["model"]
    template = runtime["template"]
    q_proj = runtime["q_proj"]
    store: LatentStore = runtime["store"]

    row = build_messages(sample, cfg)
    encoded = template.encode(row)
    batch = template.data_collator([encoded])
    batch = move_to_device(_model_inputs(batch), model_device(model))
    outputs = model(**batch, output_hidden_states=True)
    hidden = output_last_hidden_state(outputs)
    token_ids = emb_token_ids(template.tokenizer, cfg)
    states, batch_idx, _ = extract_emb_hidden_states(hidden, batch["input_ids"], token_ids)

    anchor_latent = None
    if cfg.anchor_residual:
        anchor = np.stack([store.get(sample.anchor_image_id) for _ in batch_idx.tolist()])
        anchor_latent = torch.as_tensor(anchor, device=states.device, dtype=torch.float32)
    q = q_proj(states, anchor_latent=anchor_latent).detach().cpu().numpy()
    hits = search_topk(store, q, top_k=top_k)

    slot_results = []
    for i, (slot, slot_hits) in enumerate(zip(sample.slots, hits)):
        slot_results.append(
            {
                "slot_index": i,
                "after_step": slot.after_step,
                "target_image_id": slot.image_id,
                "hits": [asdict(h) for h in slot_hits],
            }
        )
    return {"seq_id": sample.seq_id, "slots": slot_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--q-proj-dir", default=None)
    parser.add_argument("--nsr-config", default=None)
    parser.add_argument("--data", required=True, help="Trajectory JSONL.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-impl", default=None)
    args = parser.parse_args()

    if args.nsr_config:
        cfg = NSRConfig.from_file(args.nsr_config)
    elif args.adapter:
        cfg = NSRConfig.from_checkpoint(args.adapter)
    else:
        raise ValueError("--nsr-config is required when --adapter is not set")

    runtime = load_runtime(
        args.model,
        cfg,
        adapter=args.adapter,
        q_proj_dir=args.q_proj_dir,
        torch_dtype=args.torch_dtype,
        attn_impl=args.attn_impl,
    )
    out_f = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        for i, sample in enumerate(iter_jsonl(args.data)):
            if args.limit is not None and i >= args.limit:
                break
            result = retrieve_sample(sample, cfg, runtime, top_k=args.top_k)
            line = json.dumps(result, ensure_ascii=False)
            if out_f:
                out_f.write(line + "\n")
            else:
                print(line)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
