"""Small helpers for integrating NSR with ms-swift."""

from __future__ import annotations

from typing import Any

import torch

from .config import NSRConfig
from .constants import emb_token_names

NSR_EXTRA_KEYS = (
    "seq_id",
    "anchor_image_id",
    "anchor_step_index",
    "slot_image_ids",
    "slot_steps",
    "slot_step_ids",
)


def get_tokenizer(processor_or_template: Any) -> Any:
    tokenizer = getattr(processor_or_template, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor_or_template


def emb_token_ids(tokenizer: Any, cfg: NSRConfig) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(emb_token_names(cfg.num_emb_tokens))
    if not isinstance(ids, list):
        ids = [ids]
    bad = [tok for tok, tok_id in zip(emb_token_names(cfg.num_emb_tokens), ids) if tok_id is None or tok_id < 0]
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        bad += [tok for tok, tok_id in zip(emb_token_names(cfg.num_emb_tokens), ids) if tok_id == unk]
    if bad:
        raise ValueError(
            f"embedding special tokens are missing from tokenizer: {bad}. "
            "Pass --new_special_tokens with the NSR token list."
        )
    return [int(i) for i in ids]


def model_hidden_size(model: Any) -> int:
    configs = [getattr(model, "config", None)]
    for attr in ("base_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner is not None:
            configs.append(getattr(inner, "config", None))
    for cfg in configs:
        if cfg is None:
            continue
        for attr in ("hidden_size", "text_hidden_size"):
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)
        for sub_attr in ("text_config", "llm_config", "language_config"):
            sub_cfg = getattr(cfg, sub_attr, None)
            value = getattr(sub_cfg, "hidden_size", None)
            if value is not None:
                return int(value)
    raise ValueError("could not infer LLM hidden size from model config")


def output_last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is not None:
        return hidden_states[-1]
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is None and isinstance(outputs, dict):
        last_hidden = outputs.get("last_hidden_state")
    if last_hidden is not None:
        return last_hidden
    raise ValueError("model output does not contain hidden states; set output_hidden_states=True")


def model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def ensure_nested(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [[value]]
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return [list(v) for v in value]
    return [value]


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]
