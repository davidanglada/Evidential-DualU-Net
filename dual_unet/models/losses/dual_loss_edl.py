from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss_w  # expects (p_hat, y_onehot) → scalar
from .mse_loss import MSELoss

# ------------------------------
# Generic helpers
# ------------------------------

def _one_hot_from_long(y_long: torch.Tensor, K: int) -> torch.Tensor:
    return F.one_hot(y_long.long().clamp_min(0), num_classes=K).permute(0, 3, 1, 2).float()

def _label_smooth(oh: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0: 
        return oh
    K = oh.shape[1]
    return (1 - eps) * oh + eps / float(K)

def _stack_targets(key: str, targets: List[Dict], device: torch.device) -> Optional[torch.Tensor]:
    if key not in targets[0]:
        return None
    return torch.stack([t[key] for t in targets]).to(device)

def _ramp(epoch: int, T: int) -> float:
    if T <= 0:
        return 1.0
    return float(min(1.0, epoch / max(1, T)))

def _predictive_entropy_norm(p_hat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized predictive entropy in [0,1]. H(p)/logK
    p_hat: (B,K,H,W)
    """
    K = p_hat.size(1)
    p = p_hat.clamp_min(eps)
    H = -(p * p.log()).sum(dim=1)
    Hmax = float(torch.log(torch.tensor(float(K), device=p_hat.device)))
    return (H / (Hmax + 1e-12)).clamp(0.0, 1.0)

# ------------------------------
# Dirichlet KL to flat prior
# ------------------------------

def _dirichlet_kl_map(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL( Dir(alpha) || Dir(1) ) per-pixel (B,H,W). alpha >= 1.
    """
    eps = 1e-8
    alpha = alpha.clamp_min(1.0 + eps)
    K = alpha.size(1)
    S = alpha.sum(dim=1)  # (B,H,W)
    term1 = torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1)
    term2 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
    const = torch.lgamma(torch.tensor(float(K), device=alpha.device, dtype=alpha.dtype))
    return term1 - const + term2  # (B,H,W)

def _dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
    return _dirichlet_kl_map(alpha).mean()

def _sensoy_freeze_true(alpha: torch.Tensor, y_onehot_hard: torch.Tensor) -> torch.Tensor:
    """
    α̃ = y + (1 - y) ⊙ α  (freeze true-class concentration channel).
    """
    return y_onehot_hard + (1.0 - y_onehot_hard) * alpha

# ------------------------------
# EDL data terms (per-pixel maps)
# ------------------------------

def _edl_ce_map(alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    # (B,K,H,W) → (B,H,W)
    S = alpha.sum(dim=1, keepdim=True)
    data_term = torch.digamma(S) - torch.digamma(alpha)
    return (y_soft * data_term).sum(dim=1)

def _edl_mse_map(alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S.clamp_min(1e-8)
    mse = (y_soft - p) ** 2
    var = (p * (1.0 - p)) / (S + 1.0)
    return (mse + var).sum(dim=1)  # (B,H,W)

# -----------------------------------
# Dual Evidential Loss (Seg + Centroid)
# -----------------------------------

class DualLoss_Evidential(nn.Module):
    """
    Dual-task evidential loss with:
      • Seg head (Dir-K): EDL-CE|EDL-MSE (+ optional Focal Dice) + Dirichlet KL to flat prior
      • Cent head (NIG): Evidential regression on 'centroid_gaussian' map (NLL + evidence regularizer [+ optional MSE])
      • UWG for SEG head (gate or BSCE-GRA/brier style)
      • KL schedule for SEG head: constant or warm-up ramp

    Returns (loss_all, loss_seg_data, loss_cent_data, loss_dice,
             loss_kl_seg, loss_kl_cent, loss_kl_total, stats)
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,  # e.g., heavier bg for circles: [1.5,1,1,1]

        # data terms
        seg_mode: str = "edl_mse",            # kept for backward compat; unused for NIG
        weight_seg: float = 1.0,
        weight_cent: float = 1.0,
        weight_dice: float = 0.0,
        dice_gamma: float = 1.0,
        seg_label_smoothing: float = 0.0,
        ignore_index: Optional[int] = None,

        # centroid pos/neg weighting (legacy, unused with NIG regression)
        pos_weight_cent: float = 1.0,

        # KL controls (seg)
        kl_max: float = 1e-3,
        kl_schedule: str = "ramp",          # {"constant","ramp"}
        kl_ramp_epochs: int = 40,
        kl_variant_seg: str = "sensoy",     # {"sensoy","vanilla"}
        kl_apply_seg: str = "all",          # {"all","misclassified"}
        kl_boost_seg: float = 1.0,          # >1 to emphasize hard pixels in KL
        kl_entropy_gate_seg: bool = False,
        kl_entropy_thr_seg: float = 0.65,

        # UWG for SEG branch
        uwg_enable: bool = False,
        uwg_style: str = "gate",            # {"gate","brier"} ; "brier" ≈ BSCE-GRA
        uwg_proxy: str = "mse",             # {"entropy","mse"} (for gate)
        uwg_gamma: float = 10.0,            # gate steepness or brier exponent
        uwg_tau_mode: str = "median",       # {"median","fixed"}
        uwg_tau: float = 0.5,               # used if tau_mode == "fixed"
        uwg_norm_mean1: bool = True,
        eps_brier: float = 1e-8,
    ):
        super().__init__()
        assert seg_mode in {"edl_ce", "edl_mse"}  # not used, but we keep the check
        assert kl_schedule in {"constant", "ramp"}
        assert kl_variant_seg in {"sensoy", "vanilla"}
        assert kl_apply_seg in {"all", "misclassified"}
        assert uwg_style in {"gate","brier"}
        assert uwg_proxy in {"entropy","mse"}
        assert uwg_tau_mode in {"median","fixed"}

        self.K = int(num_classes)
        self.class_weights = class_weights
        self.seg_mode = seg_mode
        self.w_seg = float(weight_seg)
        self.w_cent = float(weight_cent)
        self.w_dice = float(weight_dice)
        self.dice_gamma = float(dice_gamma)

        self.seg_label_smoothing = float(seg_label_smoothing)
        self.ignore_index = ignore_index
        self.pos_weight_cent = float(pos_weight_cent)

        self.kl_max = float(kl_max)
        self.kl_schedule = kl_schedule
        self.kl_ramp_epochs = int(kl_ramp_epochs)

        self.kl_variant_seg = kl_variant_seg
        self.kl_apply_seg = kl_apply_seg
        self.kl_boost_seg = float(kl_boost_seg)
        self.kl_entropy_gate_seg = bool(kl_entropy_gate_seg)
        self.kl_entropy_thr_seg = float(kl_entropy_thr_seg)

        self.uwg_enable = bool(uwg_enable)
        self.uwg_style = uwg_style
        self.uwg_proxy = uwg_proxy
        self.uwg_gamma = float(uwg_gamma)
        self.uwg_tau_mode = uwg_tau_mode
        self.uwg_tau = float(uwg_tau)
        self.uwg_norm_mean1 = bool(uwg_norm_mean1)
        self.eps_brier = float(eps_brier)

        self._dice = DiceLoss_w(class_weights=class_weights) if self.w_dice > 0 else None

        self.mse_loss = MSELoss()

    # --------- UWG (seg branch only) ----------
    def _uwg_weights_seg(self, p_hat: torch.Tensor, y_soft: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Returns detached pixel weights (B,H,W) and tau (for gate; nan if not used).
        """
        B, K, H, W = p_hat.shape
        if not self.uwg_enable:
            return p_hat.new_ones((B, H, W)), float("nan")

        if self.uwg_style == "brier":
            # BSCE-GRA style
            brier = (p_hat - y_soft).pow(2).sum(dim=1)  # (B,H,W)
            w = (brier + self.eps_brier).pow(self.uwg_gamma).detach()
            tau_val = float("nan")
        else:
            # gate: u ∈ [0,1]
            if self.uwg_proxy == "entropy":
                u = _predictive_entropy_norm(p_hat)  # (B,H,W)
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

    # --------- Data term builders ----------
    def _seg_data_map(self, alpha: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
        m = _edl_ce_map(alpha, y_soft) if self.seg_mode == "edl_ce" else _edl_mse_map(alpha, y_soft)
        if self.class_weights is not None:
            w = self.class_weights.view(1, -1, 1, 1).to(alpha.device, alpha.dtype)
            cw = (y_soft * w).sum(dim=1)  # (B,H,W)
            m = m * cw
        return m  # (B,H,W)

    # --------- KL builders (per head) ----------
    def _kl_map_with_options(
        self,
        alpha: torch.Tensor,
        y_long: Optional[torch.Tensor],
        p_hat: torch.Tensor,
        variant: str,
        apply_on: str,
        use_entropy_gate: bool,
        entropy_thr: float
    ) -> torch.Tensor:
        if variant not in {"sensoy","vanilla"}:
            raise ValueError
        if apply_on not in {"all","misclassified"}:
            raise ValueError

        K = alpha.size(1)
        alpha_for_kl = alpha
        if (variant == "sensoy") and (y_long is not None):
            y_hard = _one_hot_from_long(y_long, K)
            alpha_for_kl = _sensoy_freeze_true(alpha, y_hard)

        kl_map = _dirichlet_kl_map(alpha_for_kl)  # (B,H,W)

        if (y_long is not None) and (apply_on == "misclassified"):
            pred = p_hat.argmax(dim=1)
            mis = (pred != y_long).float()
            kl_map = kl_map * mis

        if use_entropy_gate:
            ent = _predictive_entropy_norm(p_hat)
            hard = (ent >= entropy_thr).float()
            # emphasize where hard OR already non-zero
            kl_map = kl_map * (hard + (kl_map > 0).float()).clamp_max(1.0)

        return kl_map

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_schedule == "constant":
            return self.kl_max
        return self.kl_max * _ramp(epoch, self.kl_ramp_epochs)
    # ------------------ MAIN ------------------
    def forward(
        self,
        out: Dict[str, Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        epoch: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Expects:
          out["seg"]["alpha"], out["seg"]["p_hat"] (optional; computed if missing)
          out["cent"] with NIG fields:
              "gamma", "nu", "alpha", "beta", "S", ("y_hat" optional)
          targets:
              "segmentation_mask" or "mask_long" (long, (B,H,W))
              "centroid_gaussian" (float, (B,H,W)) from GaussianCentroidMask * centroid_scale
        """
        # ---------------- SEG BRANCH ----------------
        seg_alpha = out["seg"]["alpha"]          # (B,K,H,W)
        seg_p_hat = out["seg"].get("p_hat", None)
        if seg_p_hat is None:
            S = seg_alpha.sum(dim=1, keepdim=True)
            seg_p_hat = seg_alpha / S.clamp_min(1e-8)

        # ---------------- CENTROID BRANCH (MSE) ----------------
        pred_cent = out["cent"]
        device = seg_alpha.device

        # --- Targets ---
        y_seg_long = _stack_targets("segmentation_mask", targets, device)
        if y_seg_long is None:
            y_seg_long = _stack_targets("mask_long", targets, device)
        if y_seg_long is None:
            raise KeyError("Expected 'segmentation_mask' (or 'mask_long') in targets")

        # Centroid Gaussian map (NIG regression target)
        y_cent_gauss = _stack_targets("centroid_gaussian", targets, device)   # (B,H,W)
        if y_cent_gauss is None:
            raise KeyError("Expected 'centroid_gaussian' in targets for NIG centroid head")
        y_cent = y_cent_gauss.unsqueeze(1)   # (B,1,H,W)

        # --- Valid mask (ignore_index) ---
        if self.ignore_index is not None:
            valid = (y_seg_long != self.ignore_index).float()
        else:
            valid = torch.ones_like(y_seg_long, dtype=seg_alpha.dtype)

        # --- Smoothed seg soft labels ---
        y_seg_soft = _label_smooth(
            _one_hot_from_long(y_seg_long.clamp_min(0), self.K),
            self.seg_label_smoothing
        )

        # =======================
        # SEGMENTATION DATA TERM
        # =======================
        seg_map = self._seg_data_map(seg_alpha, y_seg_soft)   # (B,H,W)

        # UWG weights for seg
        uwg_w, uwg_tau = self._uwg_weights_seg(seg_p_hat, y_seg_soft)  # (B,H,W), scalar
        uwg_w = uwg_w * valid

        if self.uwg_enable:
            loss_seg_data = (uwg_w.detach() * seg_map * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            loss_seg_data = (seg_map * valid).sum() / valid.sum().clamp_min(1.0)

        # =======================
        # CENTROID DATA TERM (NIG regression + optional MSE)
        # =======================

        loss_cent_data = self.mse_loss(pred_cent, y_cent)

        # =======================
        # DICE (FOCAL) on seg
        # =======================
        if self.w_dice > 0 and self._dice is not None:
            if self.ignore_index is not None:
                y_seg_long_d = torch.where(
                    y_seg_long == self.ignore_index,
                    torch.zeros_like(y_seg_long),
                    y_seg_long
                )
            else:
                y_seg_long_d = y_seg_long
            y_seg_onehot_d = _one_hot_from_long(y_seg_long_d, self.K)
            dice_raw = self._dice(seg_p_hat, y_seg_onehot_d)  # 1 - dice
            if self.dice_gamma != 1.0:
                dice_raw = dice_raw.pow(self.dice_gamma)
            # keep magnitude comparable if UWG is active (optional)
            loss_dice = dice_raw * (uwg_w.detach().mean()) if self.uwg_enable else dice_raw
        else:
            loss_dice = seg_alpha.new_tensor(0.0)

        # =======================
        # KL TERMS (seg only)
        # =======================
        kl_map_seg = self._kl_map_with_options(
            seg_alpha, y_seg_long.clamp_min(0), seg_p_hat,
            variant=self.kl_variant_seg,
            apply_on=self.kl_apply_seg,
            use_entropy_gate=self.kl_entropy_gate_seg,
            entropy_thr=self.kl_entropy_thr_seg
        ) * valid

        # boost hard pixels
        w_kl_seg = torch.ones_like(kl_map_seg) + (self.kl_boost_seg - 1.0) * (kl_map_seg > 0).float()

        # only scale seg KL by UWG when style == 'gate'
        if self.uwg_enable and self.uwg_style == "gate":
            w_kl_seg = w_kl_seg * uwg_w.detach()

        loss_kl_seg_core = (kl_map_seg * w_kl_seg).sum() / valid.sum().clamp_min(1.0)

        lam = self._kl_weight(epoch)
        loss_kl_seg = lam * loss_kl_seg_core

        loss_kl_total = loss_kl_seg  # only seg contributes KL

        # =======================
        # TOTAL
        # =======================
        loss_all = (
            self.w_seg  * (loss_seg_data + self.w_dice * loss_dice)
            + self.w_cent * loss_cent_data
            + loss_kl_total
        )

        # =======================
        # STATS
        # =======================
        with torch.no_grad():
            S_seg = seg_alpha.sum(dim=1)
            conf_seg = seg_p_hat.max(dim=1).values
            denom = valid.sum().clamp_min(1.0)
            mean_S_seg     = float((S_seg * valid).sum() / denom)
            mean_max_p_seg = float((conf_seg * valid).sum() / denom)
            mis_frac       = float(((seg_p_hat.argmax(dim=1) != y_seg_long) * valid.bool()).float().sum() / denom)

            # error-based metrics (more meaningful)
            mse_cent = float(loss_cent_data)

            stats = {
                "mean_S_seg": mean_S_seg,
                "mean_max_p_seg": mean_max_p_seg,
                "misclassified_frac": mis_frac,
                "centroid_mse": mse_cent,
                "kl_weight": float(lam),
                "uwg_tau": float(uwg_tau),
                "loss_cent_mse_core": float(loss_cent_data.item()),
            }

        return (
            loss_all,
            loss_seg_data,
            loss_cent_data,
            loss_dice,
            loss_kl_seg,
            loss_kl_total,
            stats,
        )
