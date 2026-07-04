"""Special-token definitions shared by training / evaluation / inference.

Two token families:
  - ``<|slot|>``  : prompt-side marker, placed after a process step to request
    retrieval of the intermediate state at that point.
  - ``<|emb|>``   : response-side query token. Its last-layer hidden state is
    projected into the frozen DINOv3 latent space.

Numbered mode (proposal §8.3 ablation "single <emb> vs numbered <emb_i>"):
when ``num_emb_tokens > 1`` the k-th slot of a sample uses ``<|emb_k|>``
instead of the single reusable ``<|emb|>``. At most ``MAX_EMB_TOKENS`` are
supported.
"""

from __future__ import annotations

SLOT_TOKEN = "<|slot|>"
EMB_TOKEN = "<|emb|>"
MAX_EMB_TOKENS = 10


def emb_token_names(num_emb_tokens: int) -> list[str]:
    """Return the ordered list of <emb> token strings for the given mode.

    num_emb_tokens == 1  -> ["<|emb|>"]                      (single reusable)
    num_emb_tokens == N  -> ["<|emb_1|>", ..., "<|emb_N|>"]  (numbered, N<=10)
    """
    if not 1 <= num_emb_tokens <= MAX_EMB_TOKENS:
        raise ValueError(
            f"num_emb_tokens must be in [1, {MAX_EMB_TOKENS}], got {num_emb_tokens}"
        )
    if num_emb_tokens == 1:
        return [EMB_TOKEN]
    return [f"<|emb_{i}|>" for i in range(1, num_emb_tokens + 1)]


def emb_token_for_slot(slot_index: int, num_emb_tokens: int) -> str:
    """Token string used for the ``slot_index``-th (0-based) slot of a sample."""
    names = emb_token_names(num_emb_tokens)
    if num_emb_tokens == 1:
        return names[0]
    if slot_index >= num_emb_tokens:
        raise ValueError(
            f"sample has slot index {slot_index} but only {num_emb_tokens} "
            f"numbered emb tokens are configured"
        )
    return names[slot_index]


def special_token_names(num_emb_tokens: int) -> list[str]:
    """All special tokens to add to the tokenizer for the given mode."""
    return [SLOT_TOKEN] + emb_token_names(num_emb_tokens)
