"""NSRConfig — single source of truth for options that must stay consistent
across training, evaluation and inference.

The training side (ms-swift external plugin) cannot receive arbitrary CLI
flags, so the config is passed as a JSON file whose path is exported in the
``NSR_CONFIG`` environment variable. Inference / evaluation scripts accept
``--nsr_config`` directly and also fall back to the copy saved inside a
checkpoint directory (``nsr_config.json``).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .constants import MAX_EMB_TOKENS

ENV_VAR = "NSR_CONFIG"
CONFIG_FILENAME = "nsr_config.json"


@dataclass
class NSRConfig:
    # ---- special tokens -------------------------------------------------
    #: 1 → single reusable <|emb|>; N in [2,10] → numbered <|emb_1|>..<|emb_N|>
    num_emb_tokens: int = 1

    # ---- state (gallery) encoder ----------------------------------------
    #: HF id or local path of the (domain-SSL fine-tuned) DINOv3 checkpoint.
    dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    #: dimension of the DINOv3 CLS latent (768 ViT-B/16, 1024 ViT-L/16, ...).
    latent_dim: int = 768

    # ---- query head (q_proj) --------------------------------------------
    q_proj_num_layers: int = 2
    #: hidden width of the MLP; None → use the LLM hidden size.
    q_proj_hidden_dim: int | None = None
    #: q_k = norm( q_proj(h_emb) + DINOv3_CLS(anchor) )  (residual option).
    #: The model then predicts a *delta* from the anchor state instead of an
    #: absolute latent.
    anchor_residual: bool = False

    # ---- loss ------------------------------------------------------------
    temperature: float = 0.05
    lambda_lm: float = 0.5
    lambda_ret: float = 1.0
    #: mask keys with the same gallery image as the positive (false negatives).
    mask_same_image: bool = True
    #: mask keys from the same trajectory within |Δstep| <= this (0 disables).
    mask_adjacent_steps: int = 1
    #: per-slot "same step, different instance" hard negatives (0 disables).
    num_hard_negatives: int = 4

    # ---- data ------------------------------------------------------------
    #: latent store directory produced by training/precompute_latents.py.
    latent_store: str = ""
    #: prompt language: "ko" or "en".
    prompt_lang: str = "ko"
    #: anchor ablation (proposal §8.3): False drops the anchor image x_T.
    include_anchor: bool = True

    # ---- misc --------------------------------------------------------
    #: std of the noise added on top of mean-embedding init for new tokens.
    new_token_init_noise: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        if not 1 <= self.num_emb_tokens <= MAX_EMB_TOKENS:
            raise ValueError(
                f"num_emb_tokens must be in [1, {MAX_EMB_TOKENS}], "
                f"got {self.num_emb_tokens}"
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.q_proj_num_layers < 1:
            raise ValueError("q_proj_num_layers must be >= 1")

    # ---- (de)serialization ------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "NSRConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown NSRConfig keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    @classmethod
    def from_env(cls) -> "NSRConfig":
        path = os.environ.get(ENV_VAR)
        if not path:
            raise RuntimeError(
                f"environment variable {ENV_VAR} is not set; it must point to an "
                f"nsr_config.json (see configs/ for examples)"
            )
        return cls.from_file(path)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path) -> "NSRConfig":
        """Load the config copy saved next to a checkpoint."""
        path = Path(ckpt_dir) / CONFIG_FILENAME
        if not path.exists():
            # ms-swift checkpoints live in <output_dir>/checkpoint-N; the config
            # copy is written to <output_dir> by the training plugin.
            parent = Path(ckpt_dir).parent / CONFIG_FILENAME
            if parent.exists():
                path = parent
            else:
                raise FileNotFoundError(
                    f"{CONFIG_FILENAME} not found in {ckpt_dir} or its parent"
                )
        return cls.from_file(path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            path = path / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, ensure_ascii=False, indent=2)
