"""ms-swift SFT plugin for Next-State Retrieval.

Register with:

    --external_plugins training/sft/plugin.py --loss_type nsr_sft

The plugin keeps inference one-shot: the assistant response already contains
the structured <emb> tokens, so training and evaluation only need one forward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ms-swift"))

import numpy as np
import torch
import torch.nn.functional as F

from nsr.config import NSRConfig
from nsr.latent_store import LatentStore
from nsr.losses import (
    build_false_negative_mask,
    extract_emb_hidden_states,
    mask_emb_labels,
    masked_info_nce,
)
from nsr.projection import Q_PROJ_ATTR, Q_PROJ_WEIGHTS, QProjHead, load_q_proj, save_q_proj
from nsr.schema import stable_seq_hash
from nsr.swift_utils import (
    NSR_EXTRA_KEYS,
    emb_token_ids,
    ensure_list,
    ensure_nested,
    model_hidden_size,
    output_last_hidden_state,
)
from swift.callbacks.base import TrainerCallback
from swift.loss.base import BaseLoss


def _find_q_proj(model: Any) -> QProjHead | None:
    for obj in (model, getattr(model, "module", None), getattr(model, "base_model", None)):
        if obj is not None and hasattr(obj, Q_PROJ_ATTR):
            return getattr(obj, Q_PROJ_ATTR)
    return None


def _set_q_proj(model: Any, head: QProjHead) -> None:
    setattr(model, Q_PROJ_ATTR, head)


def _patch_seq2seq_trainer() -> None:
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

    if getattr(Seq2SeqTrainer, "_nsr_patched", False):
        return

    old_compute_loss = Seq2SeqTrainer.compute_loss

    def compute_loss(self, model, inputs, *args, **kwargs):
        nsr_batch = {k: inputs.pop(k) for k in NSR_EXTRA_KEYS if k in inputs}
        if nsr_batch:
            nsr_batch["input_ids"] = inputs.get("input_ids")
            inputs["output_hidden_states"] = True
        prev = getattr(self, "_nsr_batch", None)
        self._nsr_batch = nsr_batch
        try:
            return old_compute_loss(self, model, inputs, *args, **kwargs)
        finally:
            self._nsr_batch = prev

    Seq2SeqTrainer.compute_loss = compute_loss
    Seq2SeqTrainer._nsr_patched = True


class NSRSFTLoss(BaseLoss):
    """LM CE plus masked InfoNCE over <emb> hidden states."""

    def __init__(self, args, trainer):
        BaseLoss.__init__(self, args, trainer)
        self.cfg = NSRConfig.from_env()
        if not self.cfg.latent_store:
            raise ValueError("NSRConfig.latent_store must point to a precomputed latent store")
        self.store = LatentStore(self.cfg.latent_store, mmap=True)
        if self.store.dim != self.cfg.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: config={self.cfg.latent_dim}, store={self.store.dim}"
            )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.emb_token_ids = emb_token_ids(trainer.template.tokenizer, self.cfg)
        self._ensure_q_proj()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self.cfg.save(Path(args.output_dir))

    def _ensure_q_proj(self) -> QProjHead:
        head = _find_q_proj(self.trainer.model)
        if head is not None:
            return head

        hidden_size = model_hidden_size(self.trainer.model)
        init_dir = os.environ.get("NSR_Q_PROJ_INIT") or getattr(self.args, "resume_from_checkpoint", None)
        if init_dir:
            try:
                head = load_q_proj(self.cfg, hidden_size, init_dir)
            except FileNotFoundError:
                head = QProjHead.from_config(self.cfg, hidden_size)
        else:
            head = QProjHead.from_config(self.cfg, hidden_size)
        device = next(self.trainer.model.parameters()).device
        head = head.to(device=device, dtype=torch.float32)
        _set_q_proj(self.trainer.model, head)
        return head

    def _flatten_slot_metadata(self, batch: dict[str, Any]) -> tuple[list[str], list[int], list[str], list[str]]:
        slot_image_ids = ensure_nested(batch.get("slot_image_ids"))
        slot_steps = ensure_nested(batch.get("slot_steps"))
        slot_step_ids = ensure_nested(batch.get("slot_step_ids"))
        seq_ids = ensure_list(batch.get("seq_id"))

        flat_image_ids: list[str] = []
        flat_steps: list[int] = []
        flat_step_ids: list[str] = []
        flat_seq_ids: list[str] = []
        for b, images in enumerate(slot_image_ids):
            steps = slot_steps[b] if b < len(slot_steps) else []
            step_ids = slot_step_ids[b] if b < len(slot_step_ids) else []
            seq_id = str(seq_ids[b]) if b < len(seq_ids) else ""
            for s, image_id in enumerate(images):
                flat_image_ids.append(str(image_id))
                flat_steps.append(int(steps[s]) if s < len(steps) else s + 1)
                flat_step_ids.append(str(step_ids[s]) if s < len(step_ids) else str(flat_steps[-1]))
                flat_seq_ids.append(seq_id)
        return flat_image_ids, flat_steps, flat_step_ids, flat_seq_ids

    def _anchor_latents(self, batch: dict[str, Any], batch_idx: torch.Tensor, device: torch.device) -> torch.Tensor | None:
        if not self.cfg.anchor_residual:
            return None
        anchor_ids = ensure_list(batch.get("anchor_image_id"))
        if not anchor_ids:
            raise ValueError("anchor_residual=True requires anchor_image_id in the dataset")
        latents = []
        for b in batch_idx.detach().cpu().tolist():
            anchor_id = str(anchor_ids[b])
            latents.append(self.store.get(anchor_id))
        return torch.as_tensor(np.stack(latents), device=device, dtype=torch.float32)

    def _retrieval_loss(self, outputs, labels: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("NSR loss requires input_ids; inputs_embeds-only batches are unsupported")
        hidden = output_last_hidden_state(outputs)
        states, batch_idx, _ = extract_emb_hidden_states(hidden, input_ids.to(hidden.device), self.emb_token_ids)
        if states.numel() == 0:
            raise ValueError("no <emb> token positions found in batch")

        flat_image_ids, flat_steps, flat_step_ids, flat_seq_ids = self._flatten_slot_metadata(batch)
        if len(flat_image_ids) != states.shape[0]:
            raise ValueError(
                f"<emb> count ({states.shape[0]}) != slot metadata count ({len(flat_image_ids)})"
            )

        pos_gallery_idx = [self.store.index_of(image_id) for image_id in flat_image_ids]
        key_indices = list(pos_gallery_idx)
        for step_id, seq_id in zip(flat_step_ids, flat_seq_ids):
            if self.cfg.num_hard_negatives <= 0:
                continue
            hard = self.store.sample_hard_negatives(
                step_id=step_id,
                exclude_seq_id=seq_id,
                num=self.cfg.num_hard_negatives,
                rng=self.rng,
            )
            key_indices.extend(int(i) for i in hard.tolist())

        device = states.device
        keys = torch.as_tensor(
            np.asarray(self.store.latents[key_indices], dtype=np.float32),
            device=device,
            dtype=torch.float32,
        )
        keys = F.normalize(keys, dim=-1)
        anchor_latent = self._anchor_latents(batch, batch_idx, device)
        q = _find_q_proj(self.trainer.model)(states, anchor_latent=anchor_latent)

        pos_index = torch.arange(len(pos_gallery_idx), device=device, dtype=torch.long)
        key_seq_hash = torch.as_tensor(
            [stable_seq_hash(self.store.meta[i].seq_id) for i in key_indices],
            device=device,
            dtype=torch.long,
        )
        key_step = torch.as_tensor(
            [self.store.meta[i].step_index for i in key_indices],
            device=device,
            dtype=torch.long,
        )
        mask = build_false_negative_mask(
            q_image_idx=torch.as_tensor(pos_gallery_idx, device=device, dtype=torch.long),
            q_seq_hash=torch.as_tensor([stable_seq_hash(s) for s in flat_seq_ids], device=device, dtype=torch.long),
            q_step=torch.as_tensor(flat_steps, device=device, dtype=torch.long),
            key_image_idx=torch.as_tensor(key_indices, device=device, dtype=torch.long),
            key_seq_hash=key_seq_hash,
            key_step=key_step,
            pos_index=pos_index,
            mask_same_image=self.cfg.mask_same_image,
            mask_adjacent_steps=self.cfg.mask_adjacent_steps,
        )
        return masked_info_nce(q, keys, pos_index, mask, self.cfg.temperature)

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, trainer=None, **kwargs):
        from swift.trainers import per_token_loss_func

        trainer = trainer or self.trainer
        batch = getattr(trainer, "_nsr_batch", None) or {}
        labels_for_lm = mask_emb_labels(labels, self.emb_token_ids)
        token_loss = per_token_loss_func(outputs, labels_for_lm)
        if num_items_in_batch is None:
            num_items_in_batch = (labels_for_lm[:, 1:] != -100).sum().clamp_min(1)
        lm_loss = token_loss.sum() / num_items_in_batch

        if not batch or self.cfg.lambda_ret == 0:
            return self.cfg.lambda_lm * lm_loss
        ret_loss, _ = self._retrieval_loss(outputs, labels_for_lm, batch)
        return self.cfg.lambda_lm * lm_loss + self.cfg.lambda_ret * ret_loss


class NSRSaveCallback(TrainerCallback):
    def __init__(self, args, trainer):
        TrainerCallback.__init__(self, args, trainer)

    def _save(self, output_dir: str, model: Any) -> None:
        from swift.utils import is_master

        if not is_master():
            return
        head = _find_q_proj(model)
        if head is None:
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_q_proj(head, output_dir)
        try:
            NSRConfig.from_env().save(output_dir)
        except Exception:
            pass

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self._save(str(ckpt), kwargs.get("model", self.trainer.model))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save(args.output_dir, kwargs.get("model", self.trainer.model))
        return control


_patch_seq2seq_trainer()

from swift.callbacks import callbacks_map
from swift.loss import loss_map

loss_map["nsr_sft"] = NSRSFTLoss
callbacks_map["nsr_save"] = NSRSaveCallback
