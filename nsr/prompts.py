"""Prompt / response construction (proposal §3.2, §4.4, §5).

The *same* builders are used by the training preprocessor and by inference
(teacher-structured response), so the token layout seen at train and test
time is identical by construction.

Layout
------
prompt (user turn):
    <image>                               # anchor x_T (omitted for ablation)
    {instruction}
    1) step-1
    2) step-2 <|slot|>                    # slot marker after the step
    ...

response (assistant turn):
    1) step-1  2) step-2 <|emb|>  3) ...  # narration copy, <emb> at slots

Every <|emb|> sits in the response, *after* the full prompt has been
consumed, so causal attention lets each slot attend to the anchor and all
steps (proposal §4.4).
"""

from __future__ import annotations

from .config import NSRConfig
from .constants import SLOT_TOKEN, emb_token_for_slot

INSTRUCTION = {
    "ko": (
        "최종 상태 이미지와 아래 공정 순서를 참고하여, 공정 순서를 그대로 나열하되 "
        f"{SLOT_TOKEN} 로 표시된 지점의 중간 상태를 임베딩 토큰으로 예측하세요."
    ),
    "en": (
        "Given the final-state image and the process steps below, repeat the "
        f"step list and predict the intermediate state at each {SLOT_TOKEN} "
        "marker with an embedding token."
    ),
}

INSTRUCTION_NO_ANCHOR = {
    "ko": (
        "아래 공정 순서를 그대로 나열하되, "
        f"{SLOT_TOKEN} 로 표시된 지점의 중간 상태를 임베딩 토큰으로 예측하세요."
    ),
    "en": (
        "Repeat the process steps below and predict the intermediate state at "
        f"each {SLOT_TOKEN} marker with an embedding token."
    ),
}


def build_prompt(
    steps: list[str],
    slot_after: list[int],
    cfg: NSRConfig,
    include_anchor: bool | None = None,
) -> str:
    """User-turn text. ``slot_after`` holds 1-based step indices (sorted)."""
    include_anchor = cfg.include_anchor if include_anchor is None else include_anchor
    slot_set = set(slot_after)
    inst = (INSTRUCTION if include_anchor else INSTRUCTION_NO_ANCHOR)[cfg.prompt_lang]
    lines = []
    if include_anchor:
        lines.append("<image>")
    lines.append(inst)
    for i, step in enumerate(steps, start=1):
        suffix = f" {SLOT_TOKEN}" if i in slot_set else ""
        lines.append(f"{i}) {step}{suffix}")
    return "\n".join(lines)


def build_response(steps: list[str], slot_after: list[int], cfg: NSRConfig) -> str:
    """Assistant-turn text with <emb> tokens at slot positions.

    The k-th slot (in ``slot_after`` order) uses ``emb_token_for_slot(k)``:
    the single reusable <|emb|> when num_emb_tokens == 1, else <|emb_k+1|>.
    """
    slot_order = {step_idx: k for k, step_idx in enumerate(sorted(slot_after))}
    parts = []
    for i, step in enumerate(steps, start=1):
        parts.append(f"{i}) {step}")
        if i in slot_order:
            parts.append(emb_token_for_slot(slot_order[i], cfg.num_emb_tokens))
    return "  ".join(parts)
