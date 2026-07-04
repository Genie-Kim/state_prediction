"""Precomputed DINOv3 latent store (gallery) + hard-negative bank.

Directory layout (produced by ``training/precompute_latents.py``):

    <store>/
      latents.npy   float32 [N, D] raw (un-normalized) CLS latents, mmap-able
      ids.json      list[str] — image_id per row
      meta.jsonl    one row per image:
                    {"image_id", "seq_id", "step_index", "step_id", "image"}

``step_id`` is the *step-type* key (e.g. the step description or a process
code) used to sample "same step, different instance" hard negatives;
``step_index`` is the positional index inside its trajectory (for Δstep
masking / prev-state eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import stable_seq_hash

LATENTS_FILE = "latents.npy"
IDS_FILE = "ids.json"
META_FILE = "meta.jsonl"


@dataclass
class GalleryEntry:
    image_id: str
    seq_id: str
    step_index: int
    step_id: str
    image: str


class LatentStore:
    def __init__(self, root: str | Path, mmap: bool = True):
        self.root = Path(root)
        self.latents: np.ndarray = np.load(
            self.root / LATENTS_FILE, mmap_mode="r" if mmap else None
        )
        with open(self.root / IDS_FILE, encoding="utf-8") as f:
            self.ids: list[str] = json.load(f)
        if len(self.ids) != len(self.latents):
            raise ValueError(
                f"latent store corrupt: {len(self.ids)} ids vs "
                f"{len(self.latents)} latents"
            )
        self._id2idx = {im_id: i for i, im_id in enumerate(self.ids)}
        self.meta: list[GalleryEntry] = []
        with open(self.root / META_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    self.meta.append(GalleryEntry(
                        image_id=row["image_id"],
                        seq_id=row["seq_id"],
                        step_index=int(row["step_index"]),
                        step_id=str(row["step_id"]),
                        image=row["image"],
                    ))
        if len(self.meta) != len(self.ids):
            raise ValueError("latent store corrupt: meta rows != ids")
        self._step_bank: dict[str, np.ndarray] | None = None

    # ---- basic access ----------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.latents.shape[1])

    def __len__(self) -> int:
        return len(self.ids)

    def index_of(self, image_id: str) -> int:
        try:
            return self._id2idx[image_id]
        except KeyError:
            raise KeyError(
                f"image_id {image_id!r} not in latent store {self.root} "
                f"(did you run precompute_latents.py on this split?)"
            ) from None

    def get(self, image_id: str) -> np.ndarray:
        """Raw (un-normalized) float32 latent for one image."""
        return np.asarray(self.latents[self.index_of(image_id)], dtype=np.float32)

    def get_normalized(self, image_id: str) -> np.ndarray:
        z = self.get(image_id)
        return z / (np.linalg.norm(z) + 1e-12)

    # ---- metadata arrays (aligned with latent rows) ------------------
    def seq_hashes(self) -> np.ndarray:
        return np.array([stable_seq_hash(m.seq_id) for m in self.meta], dtype=np.int64)

    def step_indices(self) -> np.ndarray:
        return np.array([m.step_index for m in self.meta], dtype=np.int64)

    # ---- hard negatives ---------------------------------------------
    def _build_step_bank(self) -> dict[str, np.ndarray]:
        bank: dict[str, list[int]] = {}
        for i, m in enumerate(self.meta):
            bank.setdefault(m.step_id, []).append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in bank.items()}

    def sample_hard_negatives(
        self,
        step_id: str,
        exclude_seq_id: str,
        num: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Gallery indices of up to ``num`` same-step / different-trajectory
        images (proposal §4.5 hard negatives). May return fewer than ``num``."""
        if self._step_bank is None:
            self._step_bank = self._build_step_bank()
        pool = self._step_bank.get(step_id)
        if pool is None:
            return np.empty(0, dtype=np.int64)
        candidates = pool[[self.meta[i].seq_id != exclude_seq_id for i in pool]]
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64)
        if len(candidates) <= num:
            return candidates
        return rng.choice(candidates, size=num, replace=False)

    # ---- creation ----------------------------------------------------
    @staticmethod
    def create(
        root: str | Path,
        ids: list[str],
        latents: np.ndarray,
        meta: list[dict],
    ) -> "LatentStore":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if not (len(ids) == len(latents) == len(meta)):
            raise ValueError("ids / latents / meta length mismatch")
        np.save(root / LATENTS_FILE, np.asarray(latents, dtype=np.float32))
        with open(root / IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)
        with open(root / META_FILE, "w", encoding="utf-8") as f:
            for row in meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return LatentStore(root, mmap=True)
