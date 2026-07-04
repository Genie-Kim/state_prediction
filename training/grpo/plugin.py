"""GRPO plugin for one-shot NSR formatting rewards.

The retrieval objective is implemented in SFT because it needs hidden states at
teacher-structured <emb> positions. This GRPO plugin is intentionally limited
to one-shot format rewards: it rewards completions that emit the expected NSR
embedding-token structure without multi-turn rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ms-swift"))

from nsr.config import NSRConfig
from nsr.constants import emb_token_for_slot
from swift.rewards import ORM, orms


class NSREmbFormatReward(ORM):
    def __init__(self):
        self.cfg = NSRConfig.from_env()

    def __call__(self, completions, **kwargs: Any) -> list[float]:
        slot_steps = kwargs.get("slot_steps") or kwargs.get("slot_step_ids") or []
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            expected_slots = 1
            if i < len(slot_steps) and isinstance(slot_steps[i], (list, tuple)):
                expected_slots = len(slot_steps[i])
            expected = [
                emb_token_for_slot(slot_idx, self.cfg.num_emb_tokens)
                for slot_idx in range(expected_slots)
            ]
            present = sum(1 for token in expected if token in text)
            extra = text.count("<|emb")
            if expected_slots == 0:
                rewards.append(0.0)
            else:
                rewards.append((present / expected_slots) - max(0, extra - expected_slots) * 0.25)
        return rewards


orms["nsr_emb_format"] = NSREmbFormatReward
