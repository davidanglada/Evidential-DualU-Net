# edl_uncert_utils.py
import math
from typing import Tuple

import numpy as np
import torch
from torch.special import digamma as _digamma

# ---------- basic helpers ----------
def _safe_log(x: torch.Tensor, eps=1e-12) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))

# ---------- entropies ----------
def predictive_entropy_from_mean_lastK(p_lastK: torch.Tensor) -> torch.Tensor:
    """
    p_lastK: [..., K], probabilities per class.
    returns normalized entropy in [0,1]
    """
    K = p_lastK.size(-1)
    return (-(p_lastK * _safe_log(p_lastK)).sum(dim=-1) / math.log(K)).clamp(0.0, 1.0)

def dirichlet_expected_entropy_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    E[entropy] under Dir(alpha) normalized by log(K).
    alpha_lastK: [..., K]
    """
    K = alpha_lastK.size(-1)
    S = alpha_lastK.sum(dim=-1, keepdim=True)
    term = (alpha_lastK / S) * (_digamma(alpha_lastK + 1.0) - _digamma(S + 1.0))
    EH = -term.sum(dim=-1)
    return (EH / math.log(K)).clamp(0.0, 1.0)

def predictive_entropy_from_alpha_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    Entropy of mean categorical under Dir(alpha), normalized by log(K).
    """
    p = (alpha_lastK / alpha_lastK.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    return predictive_entropy_from_mean_lastK(p)

def dirichlet_distributional_mi_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    Mutual information = H[pred] - E[H]
    """
    Htot = predictive_entropy_from_alpha_norm(alpha_lastK)
    EH   = dirichlet_expected_entropy_norm(alpha_lastK)
    return (Htot - EH).clamp(0.0, 1.0)

# ---------- evidential UA / UE / vacuity ----------
def edl_aleatoric_scalar(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    UA = sum_k alpha_k (S - alpha_k) / [S (S+1)]
    (no normalization yet)
    """
    S = alpha_lastK.sum(dim=-1, keepdim=True)
    num = (alpha_lastK * (S - alpha_lastK)).sum(dim=-1)
    den = (S * (S + 1.0)).squeeze(-1)
    return (num / den).clamp_min(0.0)

def edl_epistemic_scalar(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    UE proxy = sum_k mu_k (1 - mu_k) / (S+1)
    """
    S  = alpha_lastK.sum(dim=-1, keepdim=True)
    mu = alpha_lastK / S
    return (mu * (1.0 - mu) / (S + 1.0)).sum(dim=-1)

def edl_vacuity(alpha_lastK: torch.Tensor) -> torch.Tensor:
    """
    Vacuity = K / S
    """
    K = alpha_lastK.size(-1)
    S = alpha_lastK.sum(dim=-1)
    return (K / S).clamp(0.0, 1.0)

# ---------- UA / UE normalization (EDL-constrained maxima) ----------
def _ua_umax(K: int) -> float: return (K - 1.0) / K
def _ue_umax(K: int) -> float: return (K - 1.0) / (K * (K + 1.0))

def normalize_ua_ue_tensor(ua: torch.Tensor, ue: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ua_max = _ua_umax(K); ue_max = _ue_umax(K)
    ua_n = (ua / max(ua_max, 1e-12)).clamp(0.0, 1.0)
    ue_n = (ue / max(ue_max, 1e-12)).clamp(0.0, 1.0)
    return ua_n, ue_n

def normalize_ua_ue_scalar(ua: float, ue: float, K: int) -> Tuple[float, float]:
    ua_max = _ua_umax(K); ue_max = _ue_umax(K)
    ua_n = float(np.clip(ua / max(ua_max, 1e-12), 0.0, 1.0))
    ue_n = float(np.clip(ue / max(ue_max, 1e-12), 0.0, 1.0))
    return ua_n, ue_n
