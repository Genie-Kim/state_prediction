"""Query projection head (proposal §4.2).

    q_k = norm( W_q h_k^{(emb)} )                        (default)
    q_k = norm( W_q h_k^{(emb)} + z_anchor )             (anchor_residual)

where ``z_anchor = DINOv3_CLS(x_T)`` (L2-normalized before the residual add),
so with the residual option the head predicts a *delta* from the anchor state
rather than an absolute latent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NSRConfig

Q_PROJ_ATTR = "nsr_q_proj"           # attribute name on the wrapped model
Q_PROJ_WEIGHTS = "nsr_q_proj.safetensors"  # side-file inside checkpoints


class QProjHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        anchor_residual: bool = False,
    ) -> None:
        super().__init__()
        self.anchor_residual = anchor_residual
        hidden_dim = hidden_dim or in_dim
        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden: torch.Tensor,                 # [S, in_dim] <emb> hidden states
        anchor_latent: torch.Tensor | None = None,  # [S, out_dim] DINOv3 CLS(x_T)
    ) -> torch.Tensor:
        param = next(self.mlp.parameters())
        hidden = hidden.to(device=param.device, dtype=param.dtype)
        q = self.mlp(hidden)
        if self.anchor_residual:
            if anchor_latent is None:
                raise ValueError(
                    "anchor_residual=True but no anchor_latent was provided"
                )
            q = q + F.normalize(anchor_latent.to(q.dtype), dim=-1)
        return F.normalize(q.float(), dim=-1)

    @classmethod
    def from_config(cls, cfg: NSRConfig, llm_hidden_size: int) -> "QProjHead":
        return cls(
            in_dim=llm_hidden_size,
            out_dim=cfg.latent_dim,
            hidden_dim=cfg.q_proj_hidden_dim,
            num_layers=cfg.q_proj_num_layers,
            anchor_residual=cfg.anchor_residual,
        )


def save_q_proj(head: QProjHead, ckpt_dir: str) -> None:
    from pathlib import Path

    from safetensors.torch import save_file

    state = {k: v.detach().float().cpu().contiguous() for k, v in head.state_dict().items()}
    save_file(state, str(Path(ckpt_dir) / Q_PROJ_WEIGHTS))


def load_q_proj(
    cfg: NSRConfig,
    llm_hidden_size: int,
    ckpt_dir: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QProjHead:
    from pathlib import Path

    from safetensors.torch import load_file

    head = QProjHead.from_config(cfg, llm_hidden_size)
    path = Path(ckpt_dir) / Q_PROJ_WEIGHTS
    if not path.exists():
        parent_path = Path(ckpt_dir).parent / Q_PROJ_WEIGHTS
        if parent_path.exists():
            path = parent_path
    if not path.exists():
        raise FileNotFoundError(f"q_proj weights not found: {path}")
    head.load_state_dict(load_file(str(path)))
    return head.to(device=device, dtype=dtype)
