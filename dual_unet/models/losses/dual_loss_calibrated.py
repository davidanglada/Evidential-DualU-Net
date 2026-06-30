import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dice_loss  import DiceLoss_w
from .brier_loss import BrierLoss          #  p∈[0,1]  version (no sigmoid)

# ────────────────────────────────────────────────────────────────
#   Dual-branch loss  (Dice + BSCE-GRA   &&   Brier on centroids)
# ────────────────────────────────────────────────────────────────
class DualLossCalibrated(nn.Module):
    """
    Segmentation branch
        • DiceLoss_w      (overlap)
        • BSCE-GRA        (Brier-weighted CE on logits)

    Centroid branch
        • BrierLoss       (proper scoring on Gaussian map, already 0-1)

    The Brier term uses the classical spatial weight
        w = 1 + λ G,   with λ clamped to a sensible interval.
    """

    def __init__(
        self,
        ce_weights: torch.Tensor,
        gamma:           float = 1.0,
        weight_dice:     float = 1.0,
        weight_ce:       float = 1.0,
        weight_brier:    float = 1.0,
        pos_weight_centroid: Optional[float] = None,
        eps:             float = 1e-6,
        λ_min:           float = 5.0,
        λ_max:           float = 40.0,
    ):
        super().__init__()
        self.gamma, self.eps   = gamma, eps
        self.w_dice            = weight_dice
        self.w_ce              = weight_ce
        self.w_brier           = weight_brier
        self.λ_min, self.λ_max = λ_min, λ_max

        self.dice_loss  = DiceLoss_w(class_weights=ce_weights)
        self.brier_loss = BrierLoss(pos_weight=pos_weight_centroid)

    # ───────── BSCE-GRA helper ─────────────────────────────────
    def _bsce_gra(self, logits: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        prob    = logits.softmax(dim=1)
        one_hot = F.one_hot(tgt_idx.long(), logits.shape[1]).permute(0,3,1,2).float()
        brier   = ((prob - one_hot) ** 2).sum(1)                     # (B,H,W)
        ce      = F.cross_entropy(logits, tgt_idx.long(),
                                  reduction="none", ignore_index=-1)
        weight  = (brier + self.eps).pow(self.gamma).detach()
        return (weight * ce).mean()

    # ───────── forward ─────────────────────────────────────────
    def forward(self, pred, target):
        pred_seg, pred_cent = pred                         # (B,C,H,W) , (B,1,H,W)
        tgt_idx   = torch.stack([t["segmentation_mask"] for t in target])
        tgt_cent  = torch.stack([t["centroid_gaussian"]   for t in target])  # (B,1,H,W)

        # ---- segmentation losses --------------------------------------
        tgt_one_hot = F.one_hot(tgt_idx.long(),
                                num_classes=pred_seg.shape[1]).permute(0,3,1,2).float()
        dice     = self.dice_loss(pred_seg.softmax(1), tgt_one_hot)
        bsce_gra = self._bsce_gra(pred_seg, tgt_idx)

        # ---- centroid weighting λ  ------------------------------------
        fg_mass = tgt_cent.sum()                           # Σ G  (real value)
        if fg_mass == 0:
            lam = 0.0                                      # all-background batch
        else:
            total_pix = float(pred_cent.numel())
            lam = (total_pix - fg_mass) / (fg_mass + self.eps)
        # clamp to a practical interval (≈ 5–40 for σ=5 px datasets)
        # lam = max(self.λ_min, min(lam, self.λ_max))
        lam = 1.0

        w_map = 1.0 + lam * tgt_cent                       # (B,1,H,W)

        # ---- Brier on centroid map ------------------------------------
        brier = self.brier_loss(pred_cent, tgt_cent.unsqueeze(1),
                                weight_map=w_map)

        # ---- total -----------------------------------------------------
        total = ( self.w_dice   * dice
                + self.w_ce     * bsce_gra
                + self.w_brier  * brier )
        return total
