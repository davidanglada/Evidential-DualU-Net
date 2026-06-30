import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

# ────────────────────────────────────────────────────────────────
# 1) Pixel‑wise BCE wrapper (works for soft Gaussian targets)
# ────────────────────────────────────────────────────────────────
class BCELoss(nn.Module):
    """Binary‑Cross‑Entropy with logits **and optional class / spatial weights**.

    Supports two typical imbalance stategies:
        • *pos_weight*  – a scalar applied to the positive class.
        • *weight_map* – per‑pixel tensor with the same shape as *pred*.

    Examples
    --------
    >>> bce = BCELoss(pos_weight=alpha)
    >>> loss = bce(pred_logits, gaussian_target)

    >>> w_map = 1.0 + 5.0 * gaussian_target            # boost peak region
    >>> loss = BCELoss()(pred_logits, gaussian_target, weight_map=w_map)
    """

    def __init__(self, pos_weight: Optional[float] = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("pos_weight", None if pos_weight is None else torch.tensor(float(pos_weight)))
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,          # raw logits  (B,1,H,W)
        target: torch.Tensor,        # soft labels  (B,1,H,W) ∈ [0,1]
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if weight_map is None:
            loss = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=self.pos_weight, reduction="mean"
            )
        else:
            # element‑wise BCE, then apply spatial weight map
            bce = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=self.pos_weight, reduction="none"
            )
            loss = (weight_map * bce).mean()
        return loss
