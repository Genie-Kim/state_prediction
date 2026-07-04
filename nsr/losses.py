"""InfoNCE with false-negative masking (proposal §4.5).

    L = -(1/S) Σ_i log  exp(<q_i, k_i+>/τ) / Σ_{j∉F_i} exp(<q_i, k_j>/τ)

Key pool per batch = every slot's positive latent (in-batch + in-sequence
negatives come for free) plus optional "same step, different instance" hard
negatives. The false-negative mask F_i removes:

  - keys backed by the same gallery image as query i's positive
    (``mask_same_image``);
  - keys from the same trajectory within ``|Δstep| <= mask_adjacent_steps``
    (near-identical neighbouring states).

The positive itself is always kept in the denominator.

All functions are pure tensor ops → unit-testable without ms-swift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_false_negative_mask(
    q_image_idx: torch.Tensor,   # [S]  gallery index of each query's positive
    q_seq_hash: torch.Tensor,    # [S]  trajectory hash of each query
    q_step: torch.Tensor,        # [S]  slot step index of each query
    key_image_idx: torch.Tensor, # [M]  gallery index of each key
    key_seq_hash: torch.Tensor,  # [M]  trajectory hash of each key
    key_step: torch.Tensor,      # [M]  step index of each key
    pos_index: torch.Tensor,     # [S]  column of the positive key per query
    mask_same_image: bool = True,
    mask_adjacent_steps: int = 1,
) -> torch.Tensor:
    """Boolean [S, M] mask; True = exclude key j from query i's denominator."""
    S, M = q_image_idx.shape[0], key_image_idx.shape[0]
    mask = torch.zeros(S, M, dtype=torch.bool, device=q_image_idx.device)
    if mask_same_image:
        mask |= q_image_idx[:, None] == key_image_idx[None, :]
    if mask_adjacent_steps > 0:
        same_seq = q_seq_hash[:, None] == key_seq_hash[None, :]
        near = (q_step[:, None] - key_step[None, :]).abs() <= mask_adjacent_steps
        mask |= same_seq & near
    # never mask the positive itself
    mask[torch.arange(S, device=mask.device), pos_index] = False
    return mask


def masked_info_nce(
    q: torch.Tensor,             # [S, D]  L2-normalized queries
    keys: torch.Tensor,          # [M, D]  L2-normalized key pool
    pos_index: torch.Tensor,     # [S]     positive column per query
    false_neg_mask: torch.Tensor | None,  # [S, M] True = exclude
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (loss, metrics). Metrics: in-batch top-1 retrieval accuracy and
    mean positive similarity — cheap training-time diagnostics."""
    logits = q @ keys.t() / temperature                     # [S, M]
    if false_neg_mask is not None:
        logits = logits.masked_fill(false_neg_mask, float("-inf"))
    loss = F.cross_entropy(logits, pos_index)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == pos_index).float().mean().item()
        pos_sim = (
            (q * keys[pos_index]).sum(-1).mean().item() if q.numel() else 0.0
        )
    return loss, {"ret_acc_inbatch": acc, "ret_pos_sim": pos_sim}


def extract_emb_hidden_states(
    hidden: torch.Tensor,        # [B, L, H] last-layer hidden states
    input_ids: torch.Tensor,     # [B, L]
    emb_token_ids: list[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather hidden states at <emb> token positions, in reading order.

    Returns (states [S, H], batch_index [S], position [S]). Rows are ordered
    (batch, position) ascending, so the s-th row of sample b corresponds to
    the s-th slot of sample b — matching per-sample slot metadata order.
    Works for both single-token and numbered-token modes.
    """
    ids = torch.as_tensor(emb_token_ids, device=input_ids.device)
    is_emb = torch.isin(input_ids, ids)                     # [B, L]
    batch_idx, pos = is_emb.nonzero(as_tuple=True)          # sorted (b, pos)
    return hidden[batch_idx, pos], batch_idx, pos


def mask_emb_labels(
    labels: torch.Tensor,        # [B, L]
    emb_token_ids: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Set labels at <emb> target positions to -100 (no CE on emission —
    proposal §4.5: the slot layout is dictated by the input, not learned)."""
    ids = torch.as_tensor(emb_token_ids, device=labels.device)
    return labels.masked_fill(torch.isin(labels, ids), -100)
