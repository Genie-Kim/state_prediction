"""Evaluate one-shot NSR retrieval with Recall@K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ms-swift"))

from inference.retrieve import load_runtime, retrieve_sample
from nsr.config import NSRConfig
from nsr.schema import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--q-proj-dir", default=None)
    parser.add_argument("--nsr-config", default=None)
    parser.add_argument("--data", required=True, help="Trajectory JSONL.")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--predictions", default=None)
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
    max_k = max(args.ks)
    counts = {k: 0 for k in args.ks}
    total = 0
    pred_f = open(args.predictions, "w", encoding="utf-8") if args.predictions else None

    try:
        for i, sample in enumerate(iter_jsonl(args.data)):
            if args.limit is not None and i >= args.limit:
                break
            result = retrieve_sample(sample, cfg, runtime, top_k=max_k)
            if pred_f:
                pred_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            for slot in result["slots"]:
                total += 1
                ranked_ids = [hit["image_id"] for hit in slot["hits"]]
                for k in args.ks:
                    counts[k] += int(slot["target_image_id"] in ranked_ids[:k])
    finally:
        if pred_f:
            pred_f.close()

    metrics = {f"recall@{k}": (counts[k] / total if total else 0.0) for k in args.ks}
    metrics["slots"] = total
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
