import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

class BrierLoss(nn.Module):
    """
    Weighted Brier score for centroid heat-maps whose *prediction is already
    in [0,1]* (no sigmoid applied inside).

    loss = mean( w * (p - target)^2 )
    """
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight))
        )

    def forward(
        self,
        prob: torch.Tensor,            # p ∈ [0,1]  (B,1,H,W)
        target: torch.Tensor,          # G ∈ [0,1]  (B,1,H,W)
        weight_map: Optional[torch.Tensor] = None
    ):
        base = (prob - target) ** 2

        if self.pos_weight is not None:
            w_cls = torch.ones_like(base)
            w_cls[target > 0] = self.pos_weight
            base = w_cls * base

        if weight_map is not None:
            base = weight_map * base

        return base.mean()