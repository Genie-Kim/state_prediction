"""nsr — Next-State Retrieval: process understanding via latent
intermediate-state retrieval (see proposal_next_state_retrieval.md).

Shared core used by training/, evaluation/ and inference/.
"""

from .config import CONFIG_FILENAME, ENV_VAR, NSRConfig
from .constants import (
    EMB_TOKEN,
    MAX_EMB_TOKENS,
    SLOT_TOKEN,
    emb_token_for_slot,
    emb_token_names,
    special_token_names,
)
from .schema import TrajectorySample, build_messages, iter_jsonl, parse_row, stable_seq_hash

__all__ = [
    "CONFIG_FILENAME",
    "ENV_VAR",
    "NSRConfig",
    "EMB_TOKEN",
    "SLOT_TOKEN",
    "MAX_EMB_TOKENS",
    "emb_token_for_slot",
    "emb_token_names",
    "special_token_names",
    "TrajectorySample",
    "build_messages",
    "iter_jsonl",
    "parse_row",
    "stable_seq_hash",
]
