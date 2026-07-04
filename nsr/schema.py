"""Trajectory JSONL schema and message construction.

One JSONL row per training example:

.. code-block:: json

    {
      "seq_id": "wafer_0001",
      "anchor": {"image": "images/wafer_0001/s6.png",
                 "image_id": "wafer_0001_s6", "step_index": 6},
      "steps": ["산화막 증착", "감광액 코팅", "노광", "현상", "식각", "감광액 제거"],
      "slots": [
        {"after_step": 2, "image": "images/wafer_0001/s2.png",
         "image_id": "wafer_0001_s2"},
        {"after_step": 4, "image": "images/wafer_0001/s4.png",
         "image_id": "wafer_0001_s4"}
      ]
    }

- ``after_step`` is the 1-based index of the step *after which* the state is
  queried (slot k in the proposal).
- ``image_id`` keys into the DINOv3 latent store (gallery).
- ``anchor.step_index`` is the anchor's timestep T (used for Δstep masking
  and prev-state evaluation).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import NSRConfig
from .prompts import build_prompt, build_response


def stable_seq_hash(seq_id: str) -> int:
    """Deterministic 63-bit non-negative hash of a sequence id (torch-safe)."""
    digest = hashlib.md5(seq_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") >> 1


@dataclass
class Slot:
    after_step: int       # 1-based step index the slot follows
    image: str            # path to the GT intermediate-state image
    image_id: str         # gallery key


@dataclass
class TrajectorySample:
    seq_id: str
    anchor_image: str
    anchor_image_id: str
    anchor_step_index: int
    steps: list[str]
    slots: list[Slot]     # sorted by after_step

    @property
    def seq_hash(self) -> int:
        return stable_seq_hash(self.seq_id)


def parse_row(row: dict[str, Any]) -> TrajectorySample:
    try:
        anchor = row["anchor"]
        slots = [
            Slot(
                after_step=int(s["after_step"]),
                image=s["image"],
                image_id=s["image_id"],
            )
            for s in row["slots"]
        ]
        sample = TrajectorySample(
            seq_id=str(row["seq_id"]),
            anchor_image=anchor["image"],
            anchor_image_id=str(anchor["image_id"]),
            anchor_step_index=int(anchor["step_index"]),
            steps=[str(s) for s in row["steps"]],
            slots=sorted(slots, key=lambda s: s.after_step),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed trajectory row (seq_id={row.get('seq_id')}): {e}") from e

    if not sample.slots:
        raise ValueError(f"trajectory {sample.seq_id} has no slots")
    n = len(sample.steps)
    for s in sample.slots:
        if not 1 <= s.after_step <= n:
            raise ValueError(
                f"trajectory {sample.seq_id}: slot after_step={s.after_step} "
                f"out of range 1..{n}"
            )
    if len({s.after_step for s in sample.slots}) != len(sample.slots):
        raise ValueError(f"trajectory {sample.seq_id}: duplicate slot positions")
    return sample


def iter_jsonl(path: str | Path) -> Iterator[TrajectorySample]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield parse_row(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e


def build_messages(
    sample: TrajectorySample,
    cfg: NSRConfig,
    include_anchor: bool | None = None,
) -> dict[str, Any]:
    """Convert a trajectory into ms-swift message format (+ slot metadata).

    Returns a dict with ``messages`` / ``images`` (standard ms-swift keys) and
    the slot ground-truth fields consumed by the training plugin.
    """
    include_anchor = cfg.include_anchor if include_anchor is None else include_anchor
    if cfg.num_emb_tokens > 1 and len(sample.slots) > cfg.num_emb_tokens:
        raise ValueError(
            f"trajectory {sample.seq_id} has {len(sample.slots)} slots but only "
            f"{cfg.num_emb_tokens} numbered emb tokens are configured"
        )
    slot_after = [s.after_step for s in sample.slots]
    out: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": build_prompt(sample.steps, slot_after, cfg, include_anchor)},
            {"role": "assistant", "content": build_response(sample.steps, slot_after, cfg)},
        ],
        "seq_id": sample.seq_id,
        "anchor_image_id": sample.anchor_image_id,
        "anchor_step_index": sample.anchor_step_index,
        "slot_image_ids": [s.image_id for s in sample.slots],
        "slot_steps": slot_after,
        "slot_step_ids": [sample.steps[s.after_step - 1] for s in sample.slots],
    }
    if include_anchor:
        out["images"] = [sample.anchor_image]
    return out
