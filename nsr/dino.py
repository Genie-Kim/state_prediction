"""Frozen DINOv3 gallery encoder (proposal §4.2).

Wraps a (domain-SSL fine-tuned) DINOv3 checkpoint via transformers AutoModel.
The state latent is the CLS representation: ``pooler_output`` when the
checkpoint provides one, else ``last_hidden_state[:, 0]``. Latents are
returned raw (un-normalized); normalization happens at use sites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Dinov3Encoder:
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = pick_device(device)
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype)
        self.model.eval().requires_grad_(False).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode_pil(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        feats: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]
            feats.append(pooled.float().cpu().numpy())
        return np.concatenate(feats, axis=0) if feats else np.empty((0, self.dim), np.float32)

    def encode_paths(
        self,
        paths: Iterable[str | Path],
        batch_size: int = 32,
    ) -> np.ndarray:
        paths = list(paths)
        feats: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            feats.append(self.encode_pil(imgs, batch_size=batch_size))
            for im in imgs:
                im.close()
        return np.concatenate(feats, axis=0) if feats else np.empty((0, self.dim), np.float32)
