"""Precompute frozen DINOv3 CLS latents for an NSR gallery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsr.dino import Dinov3Encoder
from nsr.latent_store import LatentStore
from nsr.schema import iter_jsonl


def _resolve(path: str, image_root: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else image_root / p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Trajectory JSONL.")
    parser.add_argument("--output", required=True, help="Latent store directory.")
    parser.add_argument("--dino-model", required=True, help="HF id or local DINOv3 checkpoint.")
    parser.add_argument("--image-root", default=None, help="Root for relative image paths.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    data_path = Path(args.data)
    image_root = Path(args.image_root) if args.image_root else data_path.parent
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    seen: set[str] = set()
    ids: list[str] = []
    paths: list[str] = []
    meta: list[dict] = []

    for sample in iter_jsonl(data_path):
        items = [
            (
                sample.anchor_image_id,
                sample.anchor_image,
                sample.anchor_step_index,
                sample.steps[sample.anchor_step_index - 1]
                if 1 <= sample.anchor_step_index <= len(sample.steps)
                else f"step_{sample.anchor_step_index}",
            )
        ]
        for slot in sample.slots:
            items.append(
                (
                    slot.image_id,
                    slot.image,
                    slot.after_step,
                    sample.steps[slot.after_step - 1],
                )
            )

        for image_id, image, step_index, step_id in items:
            if image_id in seen:
                continue
            seen.add(image_id)
            ids.append(image_id)
            resolved = _resolve(image, image_root)
            paths.append(resolved)
            meta.append(
                {
                    "image_id": image_id,
                    "seq_id": sample.seq_id,
                    "step_index": step_index,
                    "step_id": step_id,
                    "image": resolved,
                }
            )

    encoder = Dinov3Encoder(args.dino_model, device=args.device, dtype=dtype)
    latents = encoder.encode_paths(paths, batch_size=args.batch_size)
    store = LatentStore.create(args.output, ids, latents, meta)
    print(f"wrote {len(store)} latents to {store.root} (dim={store.dim})")


if __name__ == "__main__":
    main()
