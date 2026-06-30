import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_loss import DiceLoss_w      # weighted Dice

# ──────────────────────────────────────────────────────────────
#  Constants (must match train / eval loops)
# ──────────────────────────────────────────────────────────────
_EVIDENCE_CLIP = 10.0          # e ≤ 10  ⇒ α ≤ 11
_NIG_CLIP      = 10.0          # ν, α, β ≤ 10

# ──────────────────────────────────────────────────────────────
#  KL(Dir‖Uniform)
# ──────────────────────────────────────────────────────────────
def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    K = alpha.size(1)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    lnB_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(sum_alpha)
    lnB_uni   = -torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
    term      = ((alpha - 1) *
                 (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return (lnB_uni - lnB_alpha + term).mean()

# ──────────────────────────────────────────────────────────────
#  Dirichlet branch  (Sensoy 2018)
# ──────────────────────────────────────────────────────────────
def _dirichlet_loss(logits, target_idx, epoch, kl_anneal_epochs=10):
    C = logits.size(1)

    evidence = F.softplus(logits).clamp(max=_EVIDENCE_CLIP)
    alpha    = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True)
    p_mean   = alpha / strength

    y_onehot = F.one_hot(target_idx.clamp_min(0).long(), C).permute(0, 3, 1, 2).float()
    mask     = (target_idx != -1).unsqueeze(1)

    error    = (y_onehot - p_mean).pow(2)
    variance = p_mean * (1 - p_mean) / (strength + 1)
    sensoy   = (mask * (error + variance)).sum(1).mean()

    kl   = _kl_dirichlet_uniform(alpha * mask + 1e-6)
    lam  = min(1.0, epoch / kl_anneal_epochs)
    return sensoy + lam * kl, p_mean

# ──────────────────────────────────────────────────────────────
#  NIG branch  (Amini 2020)
# ──────────────────────────────────────────────────────────────
def _nig_loss(raw_pred, target_map, lambda_reg=5e-3):      # ← stronger penalty
    mu, nu_r, alpha_r, beta_r = torch.chunk(raw_pred, 4, dim=1)

    nu    = F.softplus(nu_r   ).clamp(max=_NIG_CLIP) + 1e-6
    alpha = F.softplus(alpha_r).clamp(max=_NIG_CLIP) + 1.0
    beta  = F.softplus(beta_r ).clamp(min=1e-4, max=_NIG_CLIP)   # ← avoid β→0
    y     = target_map

    nll = (0.5 * torch.log(math.pi / nu)
           - alpha * torch.log(beta)
           + (alpha + 0.5) *
             torch.log((y - mu).pow(2) * nu + 2 * beta * (1 + nu))
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)).mean()

    phi      = 2 * nu + alpha
    reg_term = (torch.abs(y - mu) * phi).mean()
    return nll + lambda_reg * reg_term

# ──────────────────────────────────────────────────────────────
#  Combined Evidential + Dice loss
# ──────────────────────────────────────────────────────────────
class DualLossEDL_Dice_clip(nn.Module):
    """
    Evidential DualU-Net loss
        • Dirichlet      → weight_seg
        • NIG regression → weight_reg
        • Dice overlap   → weight_dice
    """
    def __init__(
        self,
        class_weights: torch.Tensor,
        weight_seg:   float = 0.5,
        weight_reg:   float = 5.0,
        weight_dice:  float = 1.0,
        kl_anneal_epochs: int = 10,
        lambda_nig_reg:  float = 5e-3,     # ← default matches _nig_loss
    ):
        super().__init__()
        self.weight_seg  = weight_seg
        self.weight_reg  = weight_reg
        self.weight_dice = weight_dice
        self.kl_anneal_epochs = kl_anneal_epochs
        self.lambda_nig_reg   = lambda_nig_reg
        self.dice_loss = DiceLoss_w(class_weights=class_weights)

    def forward(self, pred, targets, epoch: int = 0):
        seg_logits, centroid_raw = pred
        tgt_idx  = torch.stack([t["segmentation_mask"]   for t in targets])
        tgt_cent = torch.stack([t["centroid_gaussian"]   for t in targets])

        # — Dirichlet + KL —
        loss_seg, mean_probs = _dirichlet_loss(
            seg_logits, tgt_idx, epoch, self.kl_anneal_epochs
        )

        # — Dice on mean probabilities —
        tgt_onehot = F.one_hot(tgt_idx.clamp_min(0).long(),
                               num_classes=seg_logits.size(1)) \
                     .permute(0, 3, 1, 2).float()
        dice_term = self.dice_loss(mean_probs, tgt_onehot)

        # — NIG regression —
        loss_reg = _nig_loss(centroid_raw, tgt_cent,
                             lambda_reg=self.lambda_nig_reg)

        total = (self.weight_seg  * loss_seg +
                 self.weight_reg  * loss_reg +
                 self.weight_dice * dice_term)
        return total, loss_seg, loss_reg, dice_term
