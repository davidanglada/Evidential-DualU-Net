#!/usr/bin/env python3
# seg_edl_loss.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# If you already have this, keep the import; otherwise swap for your Dice.
from .dice_loss import DiceLoss_w  # expects (p_hat, y_onehot) → scalar

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _one_hot(y_long: torch.Tensor, K: int) -> torch.Tensor:
    return F.one_hot(y_long.long().clamp_min(0), num_classes=K).permute(0, 3, 1, 2).float()

def _label_smooth(y_soft: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0: return y_soft
    K = y_soft.size(1)
    return (1.0 - eps) * y_soft + (eps / float(K))

def _stack_targets(key: str, targets: List[Dict], device: torch.device) -> Optional[torch.Tensor]:
    if key not in targets[0]:
        return None
    return torch.stack([t[key] for t in targets]).to(device)

def _ramp(epoch: int, T: int) -> float:
    if T <= 0: return 1.0
    return float(min(1.0, epoch / max(1, T)))

def predictive_entropy_norm_from_p(p_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized predictive entropy in [0,1].
    H(p) / log K ; H(p) = -sum p log p.
    """
    K = p_hat.size(1)
    p = p_hat.clamp_min(eps)
    H = -(p * p.log()).sum(dim=1)                # (B,H,W)
    Hmax = float(torch.log(torch.tensor(float(K), device=p_hat.device)))
    return (H / (Hmax + 1e-12)).clamp(0.0, 1.0)

# ------------------------------
# Dirichlet KL to flat prior
# ------------------------------
def _dirichlet_kl_to_flat_prior(alpha: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel KL( Dir(alpha) || Dir(1) ), returns (B,H,W).
    alpha: [B,K,H,W], alpha >= 1
    """
    eps = 1e-8
    alpha = alpha.clamp_min(1.0 + eps)
    K = alpha.size(1)
    S = alpha.sum(dim=1)  # (B,H,W)

    term1 = torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1)  # (B,H,W)
    term2 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
    const = torch.lgamma(torch.tensor(float(K), device=alpha.device, dtype=alpha.dtype))
    return term1 - const + term2  # (B,H,W)

def _sensoy_freeze_true(alpha: torch.Tensor, y_hard_onehot: torch.Tensor) -> torch.Tensor:
    """
    α̃ = y + (1 - y) * α  (freeze true class channel to 1, shrink others via KL).
    """
    return y_hard_onehot + (1.0 - y_hard_onehot) * alpha

# ------------------------------
# EDL data terms (per-pixel maps)
# ------------------------------
def _edl_ce_map(alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    # (B,K,H,W) → per-pixel scalar (B,H,W)
    S = alpha.sum(dim=1, keepdim=True)
    data_term = torch.digamma(S) - torch.digamma(alpha)
    return (y_soft * data_term).sum(dim=1)

def _edl_mse_map(alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    # Sensoy L2 Bayes risk (data) per pixel
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S.clamp_min(1e-8)
    mse = (y_soft - p) ** 2
    var = (p * (1.0 - p)) / (S + 1.0)
    return (mse + var).sum(dim=1)  # (B,H,W)

# -------------------------------------------------------
# SegEDL loss (UNet, seg-only) with UWG & KL controls
# -------------------------------------------------------
class SegLoss_Evidential(nn.Module):
    """
    Segmentation-only Evidential loss for UNet (color circles & beyond).

    L =  mean( w_bar ⊙ DataMap )
       + dice_w * Dice^gamma
       + λ(t) * mean( KL_map ⊙ KL_weights )

    Where:
      • DataMap = edl_ce_map or edl_mse_map
      • Dice uses predictive mean p_hat
      • KL_map = KL(Dir(α̃) || Dir(1)),
          α̃ = α (vanilla)  or  α̃ = y + (1-y)α (sensoy freeze true)
      • UWG (uncertainty-weighted gradients):
          - style "gate": w = sigmoid(γ (u - τ)), u ∈ {entropy_norm, mse_proxy}
          - style "brier": w = (Brier+ε)^γ  (BSCE-GRA-like)
        w_bar is detached and normalized to mean 1 if uwg_norm_mean1=True.
      • KL application:
          - apply in {"all","misclassified"}; optional boost on “hard” pixels,
            with entropy gate (≥ thr) if kl_entropy_gate=True.

    Returns:
      loss_all, loss_data, loss_dice, (λ * loss_kl), stats_dict
    """

    def __init__(self, *,
        seg_mode: str = "edl_ce",                  # {"edl_ce","edl_mse"}
        dice_w: float = 1.0,
        dice_gamma: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        seg_label_smoothing: float = 0.0,
        ignore_index: Optional[int] = None,

        # KL controls
        kl_variant: str = "sensoy",                # {"sensoy","vanilla"}
        kl_apply: str = "all",                     # {"all","misclassified"}
        kl_boost: float = 3.0,                     # extra weight on “hard” pixels
        kl_entropy_gate: bool = False,             # if True, OR with high-entropy mask
        kl_entropy_thr: float = 0.65,              # threshold on normalized entropy ∈ [0,1]
        kl_max: float = 1e-3,                      # λ max
        kl_ramp_epochs: int = 40,                  # warm-up epochs

        # UWG controls
        uwg_enable: bool = False,
        uwg_proxy: str = "mse",                    # {"entropy","mse"} used for 'gate'
        uwg_gamma: float = 10.0,                   # gate steepness OR brier exponent
        uwg_tau_mode: str = "median",              # {"median","fixed"} for gate
        uwg_tau: float = 0.5,                      # used if tau_mode == "fixed"
        uwg_style: str = "gate",                   # {"gate","brier"}
        uwg_norm_mean1: bool = True,               # normalize w to mean 1
        eps_brier: float = 1e-8,                   # (Brier+eps)^gamma
    ):
        super().__init__()
        assert seg_mode in {"edl_ce","edl_mse"}
        assert kl_variant in {"sensoy","vanilla"}
        assert kl_apply in {"all","misclassified"}
        assert uwg_proxy in {"entropy","mse"}
        assert uwg_tau_mode in {"median","fixed"}
        assert uwg_style in {"gate","brier"}

        self.seg_mode = seg_mode
        self.dice_w = float(dice_w)
        self.dice_gamma = float(dice_gamma)
        self.class_weights = class_weights
        self.seg_label_smoothing = float(seg_label_smoothing)
        self.ignore_index = ignore_index

        self.kl_variant = kl_variant
        self.kl_apply = kl_apply
        self.kl_boost = float(kl_boost)
        self.kl_entropy_gate = bool(kl_entropy_gate)
        self.kl_entropy_thr = float(kl_entropy_thr)
        self.kl_max = float(kl_max)
        self.kl_ramp_epochs = int(kl_ramp_epochs)

        self.uwg_enable = bool(uwg_enable)
        self.uwg_proxy = uwg_proxy
        self.uwg_gamma = float(uwg_gamma)
        self.uwg_tau_mode = uwg_tau_mode
        self.uwg_tau = float(uwg_tau)
        self.uwg_style = uwg_style
        self.uwg_norm_mean1 = bool(uwg_norm_mean1)
        self.eps_brier = float(eps_brier)

        self._dice = DiceLoss_w(class_weights=class_weights) if self.dice_w > 0 else None

    # ------------------ UWG ------------------
    def _uwg_weights(self, p_hat: torch.Tensor, y_soft: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            w_bar : (B,H,W) detached weights
            tau   : scalar (for gate); nan if not used
        """
        B, K, H, W = p_hat.shape
        if not self.uwg_enable:
            return p_hat.new_ones((B, H, W)), float("nan")

        if self.uwg_style == "brier":
            # BSCE-GRA style: w = (Brier + eps)^gamma (detach)
            brier = (p_hat - y_soft).pow(2).sum(dim=1)  # (B,H,W)
            w = (brier + self.eps_brier).pow(self.uwg_gamma).detach()
            tau_val = float("nan")
        else:
            # Gate: u ∈ [0,1] (entropy) or approximate mse proxy ∈ [0,1]
            if self.uwg_proxy == "entropy":
                u = predictive_entropy_norm_from_p(p_hat)  # (B,H,W)
            else:
                u = ((p_hat - y_soft).pow(2).sum(dim=1) / float(K)).clamp(0.0, 1.0)

            if self.uwg_tau_mode == "median":
                with torch.no_grad():
                    tau_t = torch.median(u.detach())
            else:
                tau_t = p_hat.new_tensor(self.uwg_tau)

            w = torch.sigmoid(self.uwg_gamma * (u - tau_t)).detach()
            tau_val = float(tau_t.item()) if isinstance(tau_t, torch.Tensor) else float(tau_t)

        if self.uwg_norm_mean1:
            w = w / w.mean().clamp_min(1e-8)
        return w, tau_val

    # ------------------ DATA TERM ------------------
    def _data_term(self, alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
        # Per-pixel map (B,H,W)
        if self.seg_mode == "edl_ce":
            m = _edl_ce_map(alpha, y_soft)
        else:
            m = _edl_mse_map(alpha, y_soft)

        # Optional class weights (broadcast to per-pixel via y_soft)
        if self.class_weights is not None:
            w = self.class_weights.view(1, -1, 1, 1).to(alpha.device, alpha.dtype)
            cw = (y_soft * w).sum(dim=1)  # (B,H,W)
            m = m * cw
        return m  # (B,H,W)

    # ------------------ KL TERM ------------------
    def _kl_term(self, alpha: torch.Tensor, y_long: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
        K = alpha.size(1)
        y_hard = _one_hot(y_long, K)

        alpha_for_kl = _sensoy_freeze_true(alpha, y_hard) if self.kl_variant == "sensoy" else alpha
        kl_map = _dirichlet_kl_to_flat_prior(alpha_for_kl)  # (B,H,W)

        # Mask to misclassified if requested
        if self.kl_apply == "misclassified":
            pred = p_hat.argmax(dim=1)
            mis = (pred != y_long).float()  # (B,H,W)
            kl_map = kl_map * mis

        # Entropy gate (OR with mis) to emphasize high-entropy pixels
        if self.kl_entropy_gate:
            ent = predictive_entropy_norm_from_p(p_hat)  # (B,H,W)
            hard = (ent >= self.kl_entropy_thr).float()
            kl_map = kl_map * (hard + (kl_map > 0).float()).clamp_max(1.0)

        return kl_map  # (B,H,W)

    # ------------------ MAIN ------------------
    def forward(
        self,
        out: Dict[str, torch.Tensor] | torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        epoch: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:

        # --- begin patch ---
        if isinstance(out, dict):
            if "alpha" in out:                         # UNet seg-only (preferred)
                alpha = out["alpha"]
                p_hat = out.get("p_hat", None)
            elif "seg" in out and isinstance(out["seg"], dict):  # legacy DualUNet-style nesting
                alpha = out["seg"]["alpha"]
                p_hat = out["seg"].get("p_hat", None)
            else:
                raise KeyError(f"Expected 'alpha' or nested 'seg.alpha' in model output, got keys: {list(out.keys())}")
        else:
            alpha = out
            p_hat = None
        # --- end patch ---

        if p_hat is None:
            S = alpha.sum(dim=1, keepdim=True)
            p_hat = alpha / S.clamp_min(1e-8)

        device = alpha.device
        K = alpha.size(1)

        # Targets
        y_long = _stack_targets("segmentation_mask", targets, device)
        if y_long is None:
            y_long = _stack_targets("mask_long", targets, device)
        if y_long is None:
            raise KeyError("Expected 'segmentation_mask' or 'mask_long' in targets.")

        # Ignore-index mask
        if self.ignore_index is not None:
            valid = (y_long != self.ignore_index).float()              # (B,H,W)
        else:
            valid = torch.ones_like(y_long, dtype=alpha.dtype)

        # Smoothed soft labels (for data term & UWG)
        y_soft = _label_smooth(_one_hot(y_long.clamp_min(0), K), self.seg_label_smoothing)

        # UWG weights (detached), normalized to mean~1 if requested
        w_bar, tau_val = self._uwg_weights(p_hat, y_soft)              # (B,H,W)
        # apply valid mask to weights
        w_bar = w_bar * valid

        # ---- Data term ----
        data_map = self._data_term(alpha, y_soft)                       # (B,H,W)
        # Mask out ignore pixels
        data_map = data_map * valid
        # UWG scaling (detach so it reweights gradients)
        if self.uwg_enable:
            loss_data = (w_bar.detach() * data_map).sum() / valid.sum().clamp_min(1.0)
        else:
            loss_data = data_map.sum() / valid.sum().clamp_min(1.0)

        # ---- Dice (focal) on p̂ ----
        if self.dice_w > 0 and self._dice is not None:
            # Replace ignore_index by background for Dice (or you can mask internally in your Dice)
            if self.ignore_index is not None:
                y_long_dice = torch.where(y_long == self.ignore_index, torch.zeros_like(y_long), y_long)
            else:
                y_long_dice = y_long
            y_onehot_dice = _one_hot(y_long_dice, K)
            loss_dice_raw = self._dice(p_hat, y_onehot_dice)
            if self.dice_gamma != 1.0:
                loss_dice_raw = loss_dice_raw.pow(self.dice_gamma)
            # Keep magnitude comparable if UWG is active (optional)
            loss_dice = loss_dice_raw * (w_bar.detach().mean()) if self.uwg_enable else loss_dice_raw
        else:
            loss_dice = alpha.new_tensor(0.0)

        # ---- KL term (with optional boosting & gating) ----
        kl_map = self._kl_term(alpha, y_long.clamp_min(0), p_hat)       # (B,H,W)
        # valid mask
        kl_map = kl_map * valid
        # per-pixel KL weights (boost “hard” pixels)
        # start with base: 1 for all included pixels
        kl_weights = torch.ones_like(kl_map)
        # if apply=="misclassified", kl_map already zeroed elsewhere; we still allow boost where nonzero
        # add boost factor on nonzero (heuristic): 1 + (boost-1)*I(nonzero or hard)
        kl_weights = kl_weights + (self.kl_boost - 1.0) * (kl_map > 0).float()

        # IMPORTANT: Only scale KL by UWG when style == 'gate' (to mirror gate-driven gradient shaping)
        if self.uwg_enable and self.uwg_style == "gate":
            kl_weights = kl_weights * w_bar.detach()

        loss_kl_core = (kl_map * kl_weights).sum() / valid.sum().clamp_min(1.0)
        lam = self.kl_max * _ramp(epoch, self.kl_ramp_epochs)
        loss_kl = lam * loss_kl_core

        # ---- Total ----
        loss = loss_data + self.dice_w * loss_dice + loss_kl

        # ---- Diagnostics ----
        with torch.no_grad():
            S = alpha.sum(dim=1)                         # (B,H,W)
            conf = p_hat.max(dim=1).values              # (B,H,W)
            denom = valid.sum().clamp_min(1.0)
            mean_S = float((S * valid).sum() / denom)
            mean_conf = float((conf * valid).sum() / denom)
            mis_frac = float(((p_hat.argmax(dim=1) != y_long) * valid.bool()).float().sum() / denom)

        stats = {
            "loss_data": float(loss_data.item()),
            "loss_dice": float(loss_dice.item()),
            "loss_kl":   float(loss_kl.item()),
            "kl_weight": float(lam),
            "mean_S_seg": mean_S,
            "mean_max_p_seg": mean_conf,
            "misclassified_frac": mis_frac,
            "uwg_tau": float(tau_val),
        }
        return loss, loss_data, loss_dice, loss_kl, stats
