"""Numpy retrieval helpers for precomputed DINOv3 galleries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent_store import LatentStore


@dataclass
class RetrievalHit:
    rank: int
    score: float
    image_id: str
    seq_id: str
    step_index: int
    step_id: str
    image: str


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def search_topk(
    store: LatentStore,
    queries: np.ndarray,
    top_k: int,
    chunk_size: int = 65536,
) -> list[list[RetrievalHit]]:
    """Cosine top-k over a latent store without requiring FAISS."""
    queries = l2_normalize_np(np.atleast_2d(queries))
    top_k = min(top_k, len(store))
    best_scores = np.full((queries.shape[0], top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((queries.shape[0], top_k), -1, dtype=np.int64)

    for start in range(0, len(store), chunk_size):
        end = min(start + chunk_size, len(store))
        keys = l2_normalize_np(np.asarray(store.latents[start:end], dtype=np.float32))
        scores = queries @ keys.T
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_indices = np.concatenate(
            [
                best_indices,
                np.arange(start, end, dtype=np.int64)[None, :].repeat(queries.shape[0], axis=0),
            ],
            axis=1,
        )
        part = np.argpartition(-merged_scores, kth=top_k - 1, axis=1)[:, :top_k]
        row = np.arange(queries.shape[0])[:, None]
        best_scores = merged_scores[row, part]
        best_indices = merged_indices[row, part]
        order = np.argsort(-best_scores, axis=1)
        best_scores = best_scores[row, order]
        best_indices = best_indices[row, order]

    results: list[list[RetrievalHit]] = []
    for q in range(queries.shape[0]):
        hits = []
        for rank, (idx, score) in enumerate(zip(best_indices[q], best_scores[q]), start=1):
            meta = store.meta[int(idx)]
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=float(score),
                    image_id=meta.image_id,
                    seq_id=meta.seq_id,
                    step_index=meta.step_index,
                    step_id=meta.step_id,
                    image=meta.image,
                )
            )
        results.append(hits)
    return results
