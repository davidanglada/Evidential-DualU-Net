import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss, DiceLoss_w
from .mse_loss import MSELoss


class DualLossCombinedBSCEGRA(nn.Module):
    """
    Segmentation + centroid regression loss.

    Segmentation branch:
        • DiceLoss_w  (calibration‑agnostic, for overlap quality)
        • **BSCE‑GRA** (Cross‑Entropy re‑weighted by per‑pixel Brier score
          *on gradients* – encourages calibrated probabilities)

    Centroid branch:
        • MSELoss

    Args
    ----
    ce_weights : torch.Tensor
        Class weights for Dice and CE terms.
    gamma      : float
        Exponent γ in w = (Brier + eps)^γ  (γ=1 in the paper).
    weight_dice     : float
        Weight for Dice loss.
    weight_ce       : float
        Weight for BSCE‑GRA term.
    weight_mse      : float
        Weight for centroid MSE term.
    """

    def __init__(
        self,
        ce_weights: torch.Tensor,
        gamma: float = 1.0,          # new
        weight_dice: float = 1.0,
        weight_ce: float = 1.0,
        weight_mse: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gamma, self.weight_dice, self.weight_ce, self.weight_mse, self.eps = gamma, weight_dice, weight_ce, weight_mse, eps

        self.dice_loss = DiceLoss_w(class_weights=ce_weights)
        self.ce_weights = ce_weights          # kept in case Dice uses them
        self.mse_loss = MSELoss()

    # ───────────────────────────────────────────────────────────────────
    #   BSCE‑GRA (per‑pixel version for segmentation)
    # ─────────────────────────────────────────────────────────────────────
    def _bsce_gra(self, logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """
        logits      : (B, C, H, W)
        target_idx  : (B, H, W)   int64 class indices
        """
        B, C, H, W = logits.shape
        prob = logits.softmax(dim=1)                         # (B,C,H,W)
        one_hot = torch.nn.functional.one_hot(target_idx.long(), C).permute(0, 3, 1, 2).float()

        # per‑pixel Brier score  (B, H, W)
        brier = ((prob - one_hot) ** 2).sum(1)

        # per‑pixel CE  (B, H, W)
        ce = torch.nn.functional.cross_entropy(
            logits, target_idx.long(), reduction="none", ignore_index=-1
        )

        weight = (brier + self.eps).pow(self.gamma).detach()  # detach ⇢ gradient weighting
        loss = (weight * ce).mean()
        return loss

    # ────────────────────────────────────────────────────────────────────
    def forward(self, pred, target):
        pred_seg, pred_centroids = pred                    # (B,C,H,W) , (B,1,H,W)
        tgt_idx   = torch.stack([t["segmentation_mask"] for t in target])   # (B,H,W)
        tgt_cent  = torch.stack([t["centroid_gaussian"]  for t in target])  # (B,1,H,W)

        # Dice on probabilities
        one_hot_target_segmentation = F.one_hot(
            tgt_idx.long(),
            num_classes=pred_seg.shape[1]
        ).permute(0, 3, 1, 2)
        one_hot_target_segmentation = one_hot_target_segmentation.float()

        # Dice loss on softmax probabilities for segmentation
        dice = self.dice_loss(
            torch.softmax(pred_seg, dim=1),
            one_hot_target_segmentation
        )


        # BSCE‑GRA
        bsce_gra = self._bsce_gra(pred_seg, tgt_idx)

        # centroid MSE
        mse = self.mse_loss(pred_centroids, tgt_cent.unsqueeze(1))

        total = self.weight_dice * dice + self.weight_ce * bsce_gra + self.weight_mse * mse
        return total