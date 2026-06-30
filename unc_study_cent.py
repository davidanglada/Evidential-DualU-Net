# #!/usr/bin/env python3
# edl_uncertainty_dualunet.py
#
# Evaluate EDL uncertainties (pixel + instance) for DualUNet using your dual_unet
# builders (build_model, build_dataset, build_loader) and your checkpoints.
#
# - Seg head: Dirichlet (K classes, incl. background)
# - Cent head: Beta/Dirichlet (2 classes: [centroid=1, not-centroid=0]) if present
#
# Uncertainties (all ∈ [0,1] after normalization where needed):
#   • total            = predictive entropy of mean categorical / log(K)
#   • expected         = E[ entropy ] under Dir(α) / log(K)
#   • distributional   = total - expected   (mutual information)
#   • vacuity          = K / S
#   • edl_ale          = Σ α_k(S-α_k) / [S(S+1)]  (normalized by its EDL-constrained max)
#   • edl_epi          = Σ μ_k(1-μ_k)/(S+1)       (normalized later by its EDL max)
#
# Metrics/plots:
#   • Pixel: ECE (fixed/adaptive), UCE (fixed/adaptive), AUROC_error, AURC, KS + hist/eCDF
#   • Instance: same metrics using instance-aggregated uncertainties (via watershed)
#   • Combined instance uncertainty = mean(ent_seg, ent_cent) if cent head exists
#
# Usage:
#   python edl_uncertainty_dualunet.py --config-file <cfg.yaml> [--opts key=value ...]
#
# Outputs:
#   <output_name>/uncertainty_edl/{seg_pix,cen_pix,seg_ins,cen_ins,comb_ins,viz}/...
#   <output_name>/uncertainty_edl/metrics.json

import argparse, json, math, os, os.path as osp, sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma as _digamma
from scipy.ndimage import distance_transform_edt, label as cc_label
from scipy.stats import ks_2samp
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import extrema
from skimage import exposure
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Project imports
# -------------------------
from dual_unet.utils.distributed import init_distributed_mode, is_main_process, get_rank
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader
from dual_unet.models import build_model

SIGMA_PX = 5.0                        # σ of Gaussian centroid map (px)
G_MAX    = (1.0 / (2 * math.pi * SIGMA_PX**2)) * 100.0   # analytic max of 2D Gaussian ×100

# ============================================================
# Small helpers
# ============================================================
def _resolve_ckpt_path(cfg) -> str:
    """
    Resolve a checkpoint path in this order:
      1) cfg.experiment.ckpt_path (if provided)
      2) <output_dir>/<output_name}_best.pth
      3) <output_dir>/<output_name>.pth
      4) <output_dir>/<output_name>  (as-is: commonly saved by training loop)
    """
    exp = cfg["experiment"]
    if "ckpt_path" in exp and exp["ckpt_path"]:
        return exp["ckpt_path"]

    base = osp.join(exp["output_dir"], exp["output_name"])
    candidates = [f"{base}_best.pth", f"{base}.pth", base]
    for p in candidates:
        if osp.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find a checkpoint. Tried: {', '.join(candidates)}. "
        "Provide experiment.ckpt_path in the config or ensure the file exists."
    )

def _as_one_channel_prob(p):
    """Accept [B,1,H,W] or [B,2,H,W] (use foreground channel) or fallback to first channel."""
    if p is None:
        return None
    if p.ndim == 4:
        if p.size(1) == 1:
            return p
        if p.size(1) == 2:
            return p[:, 1:2, ...]  # assume channel 1 means 'centroid=1'
        return p[:, :1, ...]
    return p

def _pick_first_present(d: dict, keys: list):
    """Return the first non-None value for any key in keys; else None."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def standardize_outputs(out):
    """
    Normalize different forward() returns into:
      seg  = {"alpha": alpha_seg, "p_hat": p_seg}
      cent = {"alpha": alpha_cent, "p_cent": p_cent} or None

    Expected DualUNet output:
      {"seg": {"alpha","p_hat"}, "cent": {"alpha","p_cent"}}
    but we’re permissive.
    """
    if not isinstance(out, dict):
        raise AssertionError("Model output must be a dict with at least the 'seg' head.")

    seg, cent = None, None

    # --- segmentation head (required) ---
    if "seg" in out and isinstance(out["seg"], dict):
        sd = out["seg"]
        alpha_s = _pick_first_present(sd, ["alpha", "alpha_seg"])
        p_s     = _pick_first_present(sd, ["p_hat", "p", "probs", "logits"])  # logits ok if model already softmaxes
        if alpha_s is None or p_s is None:
            raise AssertionError("Segmentation head missing 'alpha' or 'p_hat'/'p' in model output.")
        seg = {"alpha": alpha_s, "p_hat": p_s}
    else:
        # Some models emit flat dict for seg-only
        alpha_s = _pick_first_present(out, ["alpha", "alpha_seg"])
        p_s     = _pick_first_present(out, ["p_hat", "p", "probs", "logits"])
        if alpha_s is not None and p_s is not None:
            seg = {"alpha": alpha_s, "p_hat": p_s}

    if seg is None:
        raise AssertionError("Could not parse model output into a segmentation head (alpha, p_hat).")

    # --- centroid head (optional) ---
    if "cent" in out and isinstance(out["cent"], dict):
        cd = out["cent"]
        a_c = _pick_first_present(cd, ["alpha", "alpha_cent", "alpha_c"])
        p_c = _pick_first_present(cd, ["p_cent", "p_hat", "p", "probs"])
        if a_c is not None and p_c is not None:
            cent = {"alpha": a_c, "p_cent": _as_one_channel_prob(p_c)}

    return seg, cent


# ============================================================
# EDL utilities (last dim = classes)
# ============================================================
def _safe_log(x: torch.Tensor, eps=1e-12): return torch.log(x.clamp_min(eps))

def predictive_entropy_from_mean_lastK(p_lastK: torch.Tensor) -> torch.Tensor:
    K = p_lastK.size(-1)
    return (-(p_lastK * _safe_log(p_lastK)).sum(dim=-1) / math.log(K)).clamp(0.0, 1.0)

def edl_aleatoric_scalar(alpha_lastK: torch.Tensor) -> torch.Tensor:
    S = alpha_lastK.sum(dim=-1, keepdim=True)
    num = (alpha_lastK * (S - alpha_lastK)).sum(dim=-1)
    den = (S * (S + 1.0)).squeeze(-1)
    return (num / den).clamp_min(0.0)

def edl_epistemic_scalar(alpha_lastK: torch.Tensor) -> torch.Tensor:
    # Equivalent to edl_aleatoric/S; written in μ form for clarity
    S  = alpha_lastK.sum(dim=-1, keepdim=True)
    mu = alpha_lastK / S
    return (mu * (1.0 - mu) / (S + 1.0)).sum(dim=-1)

def edl_vacuity(alpha_lastK: torch.Tensor) -> torch.Tensor:
    K = alpha_lastK.size(-1)
    S = alpha_lastK.sum(dim=-1)
    return (K / S).clamp(0.0, 1.0)

def dirichlet_expected_entropy_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    K = alpha_lastK.size(-1)
    S = alpha_lastK.sum(dim=-1, keepdim=True)
    term = (alpha_lastK / S) * (_digamma(alpha_lastK + 1.0) - _digamma(S + 1.0))
    EH = -term.sum(dim=-1)
    return (EH / math.log(K)).clamp(0.0, 1.0)

def predictive_entropy_from_alpha_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    p = (alpha_lastK / alpha_lastK.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    return predictive_entropy_from_mean_lastK(p)

def dirichlet_distributional_mi_norm(alpha_lastK: torch.Tensor) -> torch.Tensor:
    Htot = predictive_entropy_from_alpha_norm(alpha_lastK)
    EH = dirichlet_expected_entropy_norm(alpha_lastK)
    return (Htot - EH).clamp(0.0, 1.0)

# ---- UA/UE normalization (EDL-constrained maxima) ----
def _ua_umax(K: int) -> float: return (K - 1.0) / K
def _ue_umax(K: int) -> float: return (K - 1.0) / (K * (K + 1.0))

def normalize_ua_ue_tensor(ua: torch.Tensor, ue: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ua_max = _ua_umax(K); ue_max = _ue_umax(K)
    ua_n = (ua / max(ua_max, 1e-12)).clamp(0.0, 1.0)
    ue_n = (ue / max(ue_max, 1e-12)).clamp(0.0, 1.0)
    return ua_n, ue_n

def normalize_ua_ue_scalar(ua: float, ue: float, K: int) -> Tuple[float, float]:
    ua_max = _ua_umax(K); ue_max = _ue_umax(K)
    return (float(np.clip(ua / max(ua_max, 1e-12), 0.0, 1.0)),
            float(np.clip(ue / max(ue_max, 1e-12), 0.0, 1.0)))

# ============================================================
# Watershed / pairing helpers
# ============================================================
# def _find_local_maxima(pred: np.ndarray, h: float):
#     pred_h = exposure.rescale_intensity(pred)
#     h_maxima = extrema.h_maxima(pred_h, h)
#     return h_maxima.astype(np.uint8)

# def perform_watershed(pred_mask_CxHxW: np.ndarray, seed_prob_1xHxW: np.ndarray, th_hmax: float = 0.15):
#     centroid_mask = _find_local_maxima(seed_prob_1xHxW[0], h=th_hmax)
#     _, markers = cv2.connectedComponents(centroid_mask, 4, ltype=cv2.CV_32S)

#     pred_argmax = np.argmax(pred_mask_CxHxW, axis=0).astype(np.uint8)
#     cells_mask = (pred_argmax > 0).astype(np.uint8)
#     if cells_mask.max() == 0:
#         labeled = np.zeros_like(pred_argmax, dtype=np.int32)
#         return np.zeros((0,2)), np.zeros((0,), dtype=np.int32), pred_argmax*0, (labeled>0).astype(np.uint8)

#     dist = distance_transform_edt(cells_mask)
#     w = watershed(-dist, markers, mask=cells_mask, compactness=1)
#     contOurs w = np.invert(find_boundaries(w, mode="outer", background=0))
#     w = w * contOurs w
#     binary = (w > 0).astype(np.uint8)
#     pred_mask_major = pred_argmax * binary
#     labeled, _ = cc_label(w)

#     pred_centroids, pred_classes = [], []
#     for rid in np.unique(labeled):
#         if rid == 0: continue
#         region = (labeled == rid)
#         maj = np.bincount(pred_argmax[region]).argmax()
#         pred_mask_major[region] = maj
#         coords = np.argwhere(region)
#         cyx = coords.mean(axis=0)[::-1]
#         pred_centroids.append((cyx[1], cyx[0]))  # (x,y)
#         pred_classes.append(maj)
#     return np.asarray(pred_centroids), np.asarray(pred_classes), pred_mask_major, (labeled > 0).astype(np.uint8)

def find_local_maxima(pred: np.ndarray, h: float, centers: bool = False):
    """
    Same logic as in MultiTaskEvaluationMetric.find_local_maxima.
    Returns a binary centroid_map and the list of centroid coordinates (y, x).
    """
    if not centers:
        pred_h = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred_h, h)
    else:
        h_maxima = pred

    connectivity = 4
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
        h_maxima.astype(np.uint8), connectivity, ltype=cv2.CV_32S
    )

    coords_list = []
    for i in range(num_labels):
        if i == 0:
            continue  # skip background
        coords_list.append((int(centroids[i, 1]), int(centroids[i, 0])))  # (y,x)

    centroid_map = np.zeros_like(h_maxima, dtype=np.uint8)
    kept = []
    for (r, c) in coords_list:
        centroid_map[r, c] = 255
        kept.append((r, c))

    return centroid_map, np.asarray(kept, dtype=int)


def perform_watershed(
    pred_mask_CxHxW: np.ndarray,
    pred_gauss_1xHxW: np.ndarray,
    th_hmax: float = 0.15,
):
    """
    Watershed with centroid markers — identical to MultiTaskEvaluationMetric._perform_watershed.
    Returns:
      predicted_centroids : (N,2)  (y, x) instance centroids
      predicted_classes   : (N,)   majority class per instance
      predicted_mask      : [H,W]  majority-class map (0=bg)
      cells_mask          : [H,W]  binary FG mask
    """
    # 1) seeds from Gaussian map
    centroid_mask, _ = find_local_maxima(pred_gauss_1xHxW[0], h=th_hmax)
    _, markers = cv2.connectedComponents(centroid_mask.astype(np.uint8), 4, ltype=cv2.CV_32S)

    # 2) foreground mask from seg argmax
    pred_mask_argmax = np.argmax(pred_mask_CxHxW, axis=0).astype(np.uint8)
    cells_mask = np.zeros_like(pred_mask_argmax, dtype=np.uint8)
    cells_mask[pred_mask_argmax > 0] = 1
    if cells_mask.max() == 0:
        H, W = pred_mask_argmax.shape
        return (
            np.zeros((0, 2), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((H, W), dtype=np.uint8),
            cells_mask,
        )

    # 3) watershed on distance transform
    distance_map = distance_transform_edt(cells_mask)
    watershed_result = watershed(-distance_map, markers, mask=cells_mask.astype(bool), compactness=1)

    # 4) clean thin boundaries
    contours = np.invert(find_boundaries(watershed_result, mode="outer", background=0))
    watershed_result = watershed_result * contours

    # 5) relabel instances and assign majority class + centroid
    labeled_mask, _ = cc_label(watershed_result)

    predicted_mask = np.zeros_like(pred_mask_argmax, dtype=np.uint8)
    predicted_centroids = []
    predicted_classes = []

    for rid in np.unique(labeled_mask):
        if rid == 0:
            continue
        region_mask = (labeled_mask == rid)
        class_in_region = pred_mask_argmax[region_mask]
        if class_in_region.size == 0:
            continue

        majority_class = np.bincount(class_in_region).argmax()
        predicted_mask[region_mask] = majority_class

        region_coords = np.argwhere(region_mask)  # (N,2) [row, col]
        cy, cx = region_coords.mean(axis=0)
        predicted_centroids.append((int(cy), int(cx)))
        predicted_classes.append(int(majority_class))

    predicted_centroids = (
        np.asarray(predicted_centroids, dtype=int)
        if len(predicted_centroids) > 0
        else np.zeros((0, 2), dtype=int)
    )
    predicted_classes = (
        np.asarray(predicted_classes, dtype=int)
        if len(predicted_classes) > 0
        else np.zeros((0,), dtype=int)
    )

    return predicted_centroids, predicted_classes, predicted_mask, cells_mask


# ============================================================
# Metric helpers (ECE/UCE etc.)
# ============================================================
NUM_BINS  = 15
BIN_EDGES = np.linspace(0, 1, NUM_BINS + 1)
BIN_MIDS  = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])

def _edges(bins: int = NUM_BINS): return np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)

def _ece_value(conf: np.ndarray, correct: np.ndarray, bins: int = NUM_BINS) -> float:
    edges = _edges(bins); ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if m.any():
            acc_bin  = correct[m].mean()
            conf_bin = conf[m].mean()
            ece += m.mean() * abs(acc_bin - conf_bin)
    return float(ece)

def _bin_stats_ece(conf: np.ndarray, correct: np.ndarray, bins: int = NUM_BINS):
    edges = _edges(bins)
    centers, accs, confs, counts = [], [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if m.any():
            centers.append(0.5 * (lo + hi))
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            counts.append(m.sum())
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts), edges

def _uce_value(unc: np.ndarray, correct: np.ndarray, bins: int = NUM_BINS) -> float:
    edges = _edges(bins); uce = 0.0
    err = 1.0 - correct.astype(np.float32)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (unc >= lo) & ((unc <= hi) if i == bins - 1 else (unc < hi))
        if m.any():
            err_bin = err[m].mean()
            unc_bin = unc[m].mean()
            uce += m.mean() * abs(err_bin - unc_bin)
    return float(uce)

def _bin_stats_uce(unc: np.ndarray, correct: np.ndarray, bins: int = NUM_BINS):
    edges = _edges(bins)
    centers, errs, uncs, counts = [], [], [], []
    err = 1.0 - correct.astype(np.float32)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (unc >= lo) & ((unc <= hi) if i == bins - 1 else (unc < hi))
        if m.any():
            centers.append(0.5 * (lo + hi))
            errs.append(err[m].mean())
            uncs.append(unc[m].mean())
            counts.append(m.sum())
    return np.array(centers), np.array(errs), np.array(uncs), np.array(counts), edges

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0: return np.nan
    order = np.argsort(-scores)
    y = labels[order].astype(np.int32)
    P = y.sum(); N = y.size - P
    if P == 0 or N == 0: return np.nan
    tp = fp = 0; prev_s = None
    auc = 0.0; tpr_prev, fpr_prev = 0.0, 0.0
    for i in range(y.size):
        s_val = scores[order[i]]
        if prev_s is None or s_val == prev_s:
            tp += (y[i] == 1); fp += (y[i] == 0); prev_s = s_val; continue
        tpr = tp / P; fpr = fp / N
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) * 0.5
        tpr_prev, fpr_prev = tpr, fpr
        tp += (y[i] == 1); fp += (y[i] == 0); prev_s = s_val
    tpr = tp / P; fpr = fp / N
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) * 0.5
    return float(auc)

def risk_coverage_auc(unc: np.ndarray, correct: np.ndarray) -> float:
    if unc.size == 0: return np.nan
    order = np.argsort(unc)  # accept lowest uncertainty first
    y = correct[order].astype(np.float32)
    cum = np.cumsum(y); idx = np.arange(1, y.size + 1)
    acc = cum / idx; risk = 1.0 - acc; coverage = idx / y.size
    return float(np.trapz(risk, coverage))

# ============================================================
# Adaptive ECE / UCE helpers
# ============================================================
def _normalize_confidence(scores: np.ndarray, num_classes: int | None):
    scores = np.asarray(scores, dtype=np.float64)
    if num_classes is not None and num_classes > 1:
        floor = 1.0 / float(num_classes)
        denom = (1.0 - floor)
        scores = (scores - floor) / denom if denom > 0 else scores
    return np.clip(scores, 0.0, 1.0)

def compute_ece_adaptive(scores: np.ndarray,
                         labels: np.ndarray,
                         bins: int = NUM_BINS,
                         num_classes: int | None = None):
    s = _normalize_confidence(scores, num_classes)
    y = np.asarray(labels, dtype=np.float64)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(s, q))
    if edges.size < 2: edges = np.array([0.0, 1.0])
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(s, edges, right=False) - 1, 0, len(mids) - 1)
    K = len(mids)
    cnt   = np.bincount(idx, minlength=K)
    sum_s = np.bincount(idx, weights=s, minlength=K)
    sum_y = np.bincount(idx, weights=y, minlength=K)
    nz    = cnt > 0
    conf  = np.zeros(K); conf[nz] = sum_s[nz] / cnt[nz]
    acc   = np.zeros(K); acc[nz]  = sum_y[nz] / cnt[nz]
    N = cnt.sum() if cnt.sum() else 1.0
    gaps = np.abs(acc - conf)
    ece_adapt = float((gaps * cnt).sum() / N)
    mce       = float(gaps[nz].max()) if nz.any() else 0.0
    return mids, acc, conf, cnt, ece_adapt, mce, edges

def compute_uce_adaptive(unc: np.ndarray,
                         err: np.ndarray,
                         bins: int = NUM_BINS,
                         n_classes: int | None = None,
                         normalize: bool = True):
    u = np.clip(np.asarray(unc, dtype=np.float64), 0.0, 1.0)
    e = np.clip(np.asarray(err, dtype=np.float64), 0.0, 1.0)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(u, q))
    if edges.size < 2: edges = np.array([0.0, 1.0])
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx  = np.clip(np.digitize(u, edges, right=False) - 1, 0, len(mids) - 1)
    K    = len(mids)
    cnt   = np.bincount(idx, minlength=K)
    sum_u = np.bincount(idx, weights=u, minlength=K)
    sum_e = np.bincount(idx, weights=e, minlength=K)
    nz    = cnt > 0
    u_m   = np.zeros(K); u_m[nz] = sum_u[nz] / cnt[nz]
    e_m   = np.zeros(K); e_m[nz] = sum_e[nz] / cnt[nz]
    s = 1.0 if (n_classes is None or n_classes <= 1) else (n_classes - 1.0) / n_classes
    N = cnt.sum() if cnt.sum() else 1.0
    gaps   = np.abs(e_m - s * u_m)
    raw    = float((gaps * cnt).sum() / N)
    adj    = raw / s if (normalize and s > 0) else raw
    maxuce = float((gaps[nz] / (s if (normalize and s > 0) else 1.0)).max()) if nz.any() else 0.0
    return mids, e_m, u_m, adj, maxuce, cnt, s, edges

# ============================================================
# Plot helpers
# ============================================================
def plot_reliability_adapt(
    acc, conf, cnt, ece, path, title, normalized: bool = True,
    *, x=None, edges=None, x_mode: str = "values",
    show_counts: bool = True, show_gaps: bool = False, gaps=None, mce_idx: int | None = None):
    acc  = np.asarray(acc,  dtype=float)
    conf = np.asarray(conf, dtype=float)
    cnt  = np.asarray(cnt,  dtype=float)
    K    = len(acc)
    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        if edges.size != K + 1: edges = edges[:K+1]
        x_mids = 0.5 * (edges[:-1] + edges[1:]); widths = np.diff(edges)
    elif x is not None:
        x_mids = np.asarray(x, dtype=float); widths = np.full(K, 1.0 / max(K, 1))
    elif x_mode == "quantile":
        x_mids = (np.arange(K) + 0.5) / K; widths = np.full(K, 1.0 / max(K, 1))
    else:
        x_mids = BIN_MIDS[:K]; widths = np.full(K, 1.0 / NUM_BINS)

    fig, ax1 = plt.subplots(figsize=(4, 4))
    if x_mode == "values":
        ax1.plot([0, 1], [0, 1], "k:", lw=1)
    if edges is not None:
        for e in edges[1:-1]:
            ax1.axvline(e, color="lightgray", lw=1, alpha=0.4, zorder=0)

    ax1.plot(x_mids, acc,  "o-", label="Accuracy")
    ax1.plot(x_mids, conf, "s--", label="Confidence")

    if show_gaps:
        gaps_arr = np.abs(acc - conf) if gaps is None else np.asarray(gaps, float)
        if mce_idx is None and gaps_arr.size:
            mce_idx = int(np.nanargmax(gaps_arr))
        for i in range(K):
            color = "tab:red" if (mce_idx is not None and i == mce_idx) else "gray"
            lw    = 4.0 if (mce_idx is not None and i == mce_idx) else 3.0
            y0, y1 = sorted((acc[i], conf[i]))
            ax1.vlines(x_mids[i], y0, y1, color=color, alpha=0.35, lw=lw, zorder=1)

    if x_mode == "quantile":
        xlabel = "Quantile rank"; ax1.set_xlim(0, 1)
    else:
        xlabel = "Normalized confidence [0,1]" if normalized else "Confidence"
        ax1.set_xlim(0, 1)

    ax1.set(xlabel=xlabel, ylabel="Accuracy", ylim=(0, 1))
    ax1.legend(loc="upper left")

    if show_counts:
        ax2 = ax1.twinx()
        ax2.bar(x_mids, cnt, width=widths * 0.9, color="gray", alpha=0.25, label="# samples")
        ax2.set_ylabel("Count"); ax2.legend(loc="upper right")

    if show_gaps:
        mce_val = float(np.abs(acc - conf).max()) if K else 0.0
        plt.title(f"{title}\nACE = {ece:.3f}   |   MCE = {mce_val:.3f}")
    else:
        plt.title(f"{title}\nACE = {ece:.3f}")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def plot_ece(conf_norm01, correct, bins, title, out_png):
    centers, accs, confs, counts, edges = _bin_stats_ece(conf_norm01, correct, bins)
    ece = _ece_value(conf_norm01, correct, bins)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(centers, accs, '-o', label='Accuracy')
    ax.plot(centers, confs, '--s', label='Confidence')
    ax.plot([0,1],[0,1], ':k', linewidth=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Normalized confidence [0,1]"); ax.set_ylabel("Accuracy")
    ax.set_title(f"ECE — {title}\nECE = {ece:.3f}")
    ax2 = ax.twinx(); ax2.hist(conf_norm01, bins=edges, color='gray', alpha=0.25)
    ax2.set_ylabel("Count")
    ax.legend(loc='lower right'); fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_uce(unc, correct, bins, title, out_png):
    centers, errs, uncs, counts, edges = _bin_stats_uce(unc, correct, bins)
    uce = _uce_value(unc, correct, bins)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot(centers, errs, '-o', label='Error')
    ax.plot(centers, uncs, '--s', label='Uncertainty')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Uncertainty"); ax.set_ylabel("Error")
    ax.set_title(f"UCE — {title}\nUCE = {uce:.3f}")
    ax2 = ax.twinx(); ax2.hist(unc, bins=edges, color='gray', alpha=0.25)
    ax2.set_ylabel("Count")
    ax.legend(loc='lower right'); fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_uncertainty_hist(unc, correct, title, out_png, bins=60):
    import numpy as np
    unc = np.asarray(unc)
    correct = np.asarray(correct, dtype=bool)

    finite = np.isfinite(unc)
    m_corr  = correct & finite
    m_wrong = (~correct) & finite

    # fig, ax = plt.subplots(figsize=(5.6, 3.8))
    fig, ax = plt.subplots(figsize=(4, 3))
    if m_corr.any():
        ax.hist(unc[m_corr], bins=bins, density=True, alpha=0.55, color='tab:blue', label='correct')
    if m_wrong.any():
        ax.hist(unc[m_wrong], bins=bins, density=True, alpha=0.55, color='tab:red', label='wrong')

    ax.set_xlim(0,1); ax.set_xlabel("Uncertainty"); ax.set_ylabel("Density")
    ax.set_title(title)
    if m_corr.any() or m_wrong.any():
        ax.legend()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_ecdf(unc, correct, title, out_png):
    def _ecdf(x):
        x = np.sort(x); y = np.arange(1, len(x)+1) / float(len(x))
        return x, y
    if correct.any() and (~correct).any():
        ks = ks_2samp(unc[correct], unc[~correct]).statistic
    else:
        ks = np.nan
    uc, yc = _ecdf(unc[correct]) if correct.any() else (np.array([0.]), np.array([0.]))
    uw, yw = _ecdf(unc[~correct]) if (~correct).any() else (np.array([0.]), np.array([0.]))
    # fig, ax = plt.subplots(figsize=(5.6, 5.6))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(uc, yc, color='tab:blue', label='correct'); ax.plot(uw, yw, color='tab:red', label='wrong')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("Uncertainty threshold"); ax.set_ylabel("eCDF")
    ax.set_title(f"eCDF — {title} \nmax Δ = {ks:.3f}" if not np.isnan(ks) else "")
    ax.grid(ls=":"); ax.legend(loc='lower right'); fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)

def _subsample_indices(n: int, max_points: int = 20000, rng: np.random.Generator | None = None):
    """Return indices to subsample up to max_points points uniformly at random."""
    if n <= max_points:
        return np.arange(n)
    if rng is None:
        rng = np.random.default_rng(0)
    return rng.choice(n, size=max_points, replace=False)


def plot_2d_seg_cent_uncert(
    seg_u_ins: Dict[str, np.ndarray],
    cen_u_ins: Dict[str, np.ndarray],
    seg_corr_ins: np.ndarray,
    cen_corr_ins: np.ndarray,
    out_root: Path,
    max_points: int = 20000,
):
    """
    For each seg-uncert type (edl_ale, edl_epi, vacuity) and cent-uncert type
    (peak, mass, shift, all), plot a 2D scatter:
        x = seg uncertainty
        y = centroid uncertainty
    Points are colored by 4 correctness groups:
        0: seg_correct & cen_correct
        1: seg_wrong   & cen_correct
        2: seg_correct & cen_wrong
        3: seg_wrong   & cen_wrong
    """
    seg_keys  = ["edl_ale", "edl_epi", "vacuity"]
    cent_keys = ["peak", "mass", "shift", "all"]

    # basic sanity
    if seg_corr_ins.size == 0 or cen_corr_ins.size == 0:
        print("(Note) 2D seg-cent scatter skipped: empty seg_corr_ins or cen_corr_ins.")
        return

    out_dir = out_root / "scatter_2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    for s_key in seg_keys:
        if s_key not in seg_u_ins:
            print(f"(Note) seg uncertainty '{s_key}' not found, skipping.")
            continue
        u_seg_full = np.asarray(seg_u_ins[s_key], dtype=np.float32)

        for c_key in cent_keys:
            if c_key not in cen_u_ins:
                print(f"(Note) cent uncertainty '{c_key}' not found, skipping.")
                continue
            u_cent_full = np.asarray(cen_u_ins[c_key], dtype=np.float32)

            # Align lengths
            n = min(u_seg_full.size, u_cent_full.size, seg_corr_ins.size, cen_corr_ins.size)
            if n == 0:
                continue

            u_seg  = u_seg_full[:n]
            u_cent = u_cent_full[:n]
            s_corr = seg_corr_ins[:n].astype(bool)
            c_corr = cen_corr_ins[:n].astype(bool)

            # finite mask
            finite = np.isfinite(u_seg) & np.isfinite(u_cent)
            if not finite.any():
                continue

            u_seg  = u_seg[finite]
            u_cent = u_cent[finite]
            s_corr = s_corr[finite]
            c_corr = c_corr[finite]

            n_eff = u_seg.size
            if n_eff == 0:
                continue

            # subsample to avoid huge plots
            idx = _subsample_indices(n_eff, max_points=max_points, rng=rng)
            u_seg  = u_seg[idx]
            u_cent = u_cent[idx]
            s_corr = s_corr[idx]
            c_corr = c_corr[idx]

            # groups
            g0 = s_corr & c_corr            # both correct
            g1 = (~s_corr) & c_corr         # seg wrong, cent correct
            g2 = s_corr & (~c_corr)         # seg correct, cent wrong
            g3 = (~s_corr) & (~c_corr)      # both wrong

            groups = [g0, g1, g2, g3]
            labels = [
                "seg✔ / cent✔",
                "seg✘ / cent✔",
                "seg✔ / cent✘",
                "seg✘ / cent✘",
            ]

            colors = ["tab:green", "tab:orange", "tab:blue", "tab:red"]

            # plot
            fig, ax = plt.subplots(figsize=(6, 6))
            for mask, lab, col in zip(groups, labels, colors):
                if mask.any():
                    ax.scatter(
                        u_seg[mask],
                        u_cent[mask],
                        s=10,
                        alpha=0.4,
                        label=f"{lab} (n={mask.sum()})",
                        edgecolors="none",
                        color=col,
                    )

            ax.set_xlabel(f"Segmentation uncertainty: {s_key}")
            ax.set_ylabel(f"Centroid uncertainty: {c_key}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            title = f"2D uncertainty — seg: {s_key} vs cent: {c_key}"
            ax.set_title(title)
            ax.legend(loc="best", fontsize=8)

            fname = out_dir / f"scatter_{s_key}_vs_{c_key}.png"
            fig.tight_layout()
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            
def plot_2d_detection_uncert(
    seg_u_ins: Dict[str, np.ndarray],
    cen_u_ins: Dict[str, np.ndarray],
    det_corr_ins: np.ndarray,
    out_root: Path,
    max_points: int = 20000,
):
    """
    Plot 2D scatter plots of segmentation vs centroid uncertainty,
    but ONLY with detection correctness:

        GREEN = correct detection (paired + correct class)
        RED   = wrong detection (unmatched or wrong class)

    Inputs:
        seg_u_ins: dict { "edl_ale": [N], "edl_epi": [N], "vacuity": [N], ... }
        cen_u_ins: dict { "peak": [N], "mass": [N], "shift": [N], "all": [N], ... }
        det_corr_ins: boolean array [N], True = correct detection
        out_root: where to save plots
        max_points: subsampling limit to keep plots readable
    """
    seg_keys  = ["edl_ale", "edl_epi", "vacuity"]
    cent_keys = ["peak", "mass", "shift", "all"]

    if det_corr_ins.size == 0:
        print("(Note) 2D detection-only scatter skipped: empty det_corr_ins.")
        return

    out_dir = out_root / "scatter_2d_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    for s_key in seg_keys:
        if s_key not in seg_u_ins:
            print(f"(Note) seg uncertainty '{s_key}' not found, skipping.")
            continue
        u_seg_full = np.asarray(seg_u_ins[s_key], dtype=np.float32)

        for c_key in cent_keys:
            if c_key not in cen_u_ins:
                print(f"(Note) cent uncertainty '{c_key}' not found, skipping.")
                continue
            u_cent_full = np.asarray(cen_u_ins[c_key], dtype=np.float32)

            # Align lengths
            n = min(u_seg_full.size, u_cent_full.size, det_corr_ins.size)
            if n == 0:
                continue

            u_seg  = u_seg_full[:n]
            u_cent = u_cent_full[:n]
            d_corr = det_corr_ins[:n].astype(bool)

            # Keep only finite entries
            finite = np.isfinite(u_seg) & np.isfinite(u_cent)
            if not finite.any():
                continue

            u_seg  = u_seg[finite]
            u_cent = u_cent[finite]
            d_corr = d_corr[finite]

            n_eff = u_seg.size
            if n_eff == 0:
                continue

            # Subsample for readability
            if n_eff > max_points:
                idx = rng.choice(n_eff, size=max_points, replace=False)
                u_seg  = u_seg[idx]
                u_cent = u_cent[idx]
                d_corr = d_corr[idx]

            # Split groups
            g_correct = d_corr
            g_wrong   = ~d_corr

            # Plot
            fig, ax = plt.subplots(figsize=(6, 6))

            # Correct detections
            if g_correct.any():
                ax.scatter(
                    u_seg[g_correct],
                    u_cent[g_correct],
                    s=10,
                    alpha=0.4,
                    color="tab:green",
                    edgecolors="none",
                    label=f"correct (n={g_correct.sum()})",
                )

            # Wrong detections
            if g_wrong.any():
                ax.scatter(
                    u_seg[g_wrong],
                    u_cent[g_wrong],
                    s=10,
                    alpha=0.4,
                    color="tab:red",
                    edgecolors="none",
                    label=f"errors (n={g_wrong.sum()})",
                )

            ax.set_xlabel(f"Seg uncertainty: {s_key}")
            ax.set_ylabel(f"Cent uncertainty: {c_key}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            title = f"DETECTION-only 2D uncertainty — seg({s_key}) vs cent({c_key})"
            ax.set_title(title)
            ax.legend(loc="best", fontsize=9)

            fname = out_dir / f"scatter_det_{s_key}_vs_{c_key}.png"
            fig.tight_layout()
            fig.savefig(fname, dpi=200)
            plt.close(fig)


# ============================================================
# Main uncertainty runner
# ============================================================
@torch.no_grad()
def run_uncertainty(cfg: Dict):
    # --- device & seed / dist ---
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg['experiment']['seed'] + get_rank())

    # --- outputs ---
    out_root = Path(cfg['experiment']['output_name']) / "uncertainty_edl_mse_plots_rebuttal"
    for d in ["seg_pix", "cen_pix", "seg_ins", "cen_ins", "comb_ins", "viz"]:
        (out_root / d).mkdir(parents=True, exist_ok=True)

    # --- data/model ---
    test_dataset = build_dataset(cfg, split='test')
    test_loader  = build_loader(cfg, test_dataset, split='test')
    model = build_model(cfg).to(device)

    # --- load checkpoint ---
    ckpt_path = _resolve_ckpt_path(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] {ckpt_path}\n  missing={len(missing)}  unexpected={len(unexpected)}")

    model.eval()

    # --- params ---
    bins = int(cfg['evaluation'].get('bins', NUM_BINS))
    hmax_th  = float(cfg['evaluation'].get('centroid_h', 0.15))
    cent_thr = float(cfg['evaluation'].get('centroid_thr', 0.15))

    # NEW: how many images to process
    n_images_cfg = cfg['evaluation'].get('n_images', 'all')
    process_all = (isinstance(n_images_cfg, str) and n_images_cfg.lower() == 'all')
    n_images = None if process_all else int(n_images_cfg)
    processed = 0
    # --- accumulators  (pixel-level) ---
    seg_conf_pix, seg_corr_pix = [], []
    seg_u_pix = {n: [] for n in ["edl_ale", "edl_epi", "vacuity", "total", "expected", "distributional"]}

    # ### NEW: cell-union pixel accumulators
    seg_conf_pix_cell, seg_corr_pix_cell = [], []
    seg_u_pix_cell = {n: [] for n in ["edl_ale", "edl_epi", "vacuity", "total", "expected", "distributional"]}

    # --- accumulators  (instance-level) ---
    seg_conf_ins, seg_corr_ins, seg_ent_ins = [], [], []
    seg_u_ins = {n: [] for n in ["edl_ale", "edl_epi", "vacuity", "total", "expected", "distributional"]}

    cen_conf_ins, cen_corr_ins = [], []
    u_cent_ins = {n: [] for n in ["peak", "mass", "shift", "all"]}

    printed_softmax_check = False
    K_seg = None
    K_fg  = None

    for images, targets in test_loader:
        if not process_all and processed >= n_images:
            break
        images = images.to(device, non_blocking=True)
        targets = [{k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in t.items()} for t in targets]

        out = model(images)
        seg, cent = standardize_outputs(out)
        cent = out["cent"]
        have_cent = (cent is not None)

        # ------ segmentation head ------
        alpha_s = seg["alpha"]; p_s = seg["p_hat"]         # [B,K,H,W] with K incl. background at index 0
        B, K, H, W = p_s.shape

        # decide how many items from this batch to use
        if process_all:
            B_eff = B
        else:
            remaining = n_images - processed
            if remaining <= 0:
                break
            B_eff = min(B, remaining)



        if K_seg is None:
            K_seg = K
            K_fg  = max(K_seg - 1, 1)
            print(f"[INFO] K={K_seg} (incl. bg); K_fg={K_fg} for FG-only eval.")
        else:
            assert K == K_seg, f"Model seg head K={K} but previously set K_seg={K_seg}."

        if not printed_softmax_check:
            p_sum = p_s.sum(dim=1)
            print(f"[CHECK] sum_k p over classes: min={float(p_sum.min()):.6f} max={float(p_sum.max()):.6f} (≈1)")
            printed_softmax_check = True

        # Predictions and confidences
        pred_all = p_s.argmax(dim=1)                      # [B,H,W] over ALL K
        conf_all = p_s.max(dim=1).values                  # [B,H,W] over ALL K
        conf_fg  = p_s[:, 1:, ...].max(dim=1).values      # [B,H,W] over FOREGROUND only

        # last-dim = classes (for formulas)
        al_all = alpha_s.permute(0,2,3,1).contiguous()    # [B,H,W,K]
        p_all  = p_s.permute(0,2,3,1).contiguous()        # [B,H,W,K]
        al_fg  = al_all[..., 1:]                          # [B,H,W,K_fg]  (DROP BACKGROUND)

        # ---------- PIXEL-LEVEL (ALL classes, include background) ----------
        # Use K_seg (incl. background) for UA/UE normalization & entropies
        ua_all_raw = edl_aleatoric_scalar(al_all)                      # [B,H,W]
        ue_all_raw = edl_epistemic_scalar(al_all)                      # [B,H,W]
        ua_all, ue_all = normalize_ua_ue_tensor(ua_all_raw, ue_all_raw, K_seg)

        S_all   = al_all.sum(dim=-1)                                   # [B,H,W]
        vac_all = (K_seg / S_all).clamp(0.0, 1.0)                      # [B,H,W]

        # Entropy family with ALL-K (includes bg)
        tot_all = predictive_entropy_from_alpha_norm(al_all)            # [B,H,W] in [0,1]
        exp_all = dirichlet_expected_entropy_norm(al_all)               # [B,H,W] in [0,1]
        dis_all = (tot_all - exp_all).clamp(0.0, 1.0)

        u_pix_all = {
            "edl_ale": ua_all,
            "edl_epi": ue_all,
            "vacuity": vac_all,
            "total":   tot_all,
            "expected": exp_all,
            "distributional": dis_all,
        }

        # ------ iterate items in batch ------
        for i in range(B_eff):
            tgt = targets[i]
            gt_sem   = tgt["segmentation_mask"].long()                 # [H,W], 0..K-1
            all_mask = torch.ones_like(gt_sem, dtype=torch.bool)       # include background too

            # Pixel-level: ALL-K (incl. background)
            seg_conf_pix.append(conf_all[i][all_mask].detach().cpu().numpy())

            seg_corr_pix.append((pred_all[i][all_mask] == gt_sem[all_mask]).detach().cpu().numpy())

            for key in seg_u_pix:
                seg_u_pix[key].append(u_pix_all[key][i][all_mask].detach().cpu().numpy())

            # ### NEW: cell-union mask = (GT>0) ∪ (Pred>0)
            cell_union_mask = ((gt_sem > 0) | (pred_all[i] > 0))
            if cell_union_mask.any():
                seg_conf_pix_cell.append(conf_all[i][cell_union_mask].detach().cpu().numpy())
                seg_corr_pix_cell.append((pred_all[i][cell_union_mask] == gt_sem[cell_union_mask]).detach().cpu().numpy())
                for key in seg_u_pix_cell:
                    seg_u_pix_cell[key].append(u_pix_all[key][i][cell_union_mask].detach().cpu().numpy())



            # ---- INSTANCE-LEVEL via watershed on predictions ----
            pred_mask_C = p_s[i].detach().cpu().numpy()      # [K,H,W]
            seed_prob_1 = cent[i].detach().cpu().numpy()   # [1,H,W] (true or proxy)
            pred_centroids, pred_classes, pred_major, cells_mask = perform_watershed(
                pred_mask_C, seed_prob_1, th_hmax=hmax_th
            )

            labeled, _ = cc_label((pred_major > 0).astype(np.uint8))
            # Optional visualization overlay
            img_uint8 = (images[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            color = np.zeros_like(img_uint8)
            # Build a simple HSV wheel for K_seg-1 foreground classes + black for bg
            cols = [(0,0,0)]  # bg = black
            if K_seg <= 12:
                # evenly spaced hues
                for j in range(1, K_seg):
                    h = int(180 * (j-1) / max(K_seg-1,1))  # 0..179 in OpenCV HSV
                    hsv = np.uint8([[[h, 200, 255]]])      # vivid-ish
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
                    cols.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))  # to RGB
            else:
                # fallback: repeating distinct colors
                base = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
                cols += base * ((K_seg-1 + len(base)-1)//len(base))

            for cls_id in range(K_seg):
                color[pred_major == cls_id] = cols[cls_id]

            overlay = (0.55 * img_uint8 + 0.45 * color).astype(np.uint8)
            fname = Path(tgt.get("file_name", tgt.get("name", f"img_{i}"))).stem
            # cv2.imwrite(str(out_root / "viz" / f"{fname}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # ---- aggregate per instance ----
            for rid in np.unique(labeled):
                if rid == 0: continue
                region = (labeled == rid)
                if not region.any(): continue

                # -------- SEG instance (foreground classes only) --------
                if K_seg > 1:
                    # Drop background (index 0), aggregate alpha, THEN normalize → probabilities
                    al_pix_fg  = alpha_s[i, 1:, region]          # [K_fg, P]
                    al_inst_fg = al_pix_fg.mean(dim=1)           # [K_fg]
                    S_fg       = al_inst_fg.sum().clamp_min(1e-8)
                    p_inst_fg  = (al_inst_fg / S_fg)             # [K_fg]

                    # DEBUG: print the actual vectors & sums (helps catch bg leakage)
                    al_vec = al_inst_fg.detach().cpu().numpy()
                    p_vec  = p_inst_fg.detach().cpu().numpy()

                    ent = predictive_entropy_from_mean_lastK(p_inst_fg.unsqueeze(0)).item()
                    exp = dirichlet_expected_entropy_norm(al_inst_fg.unsqueeze(0)).item()
                    dis = max(0.0, ent - exp)
                    ua_inst_raw = edl_aleatoric_scalar(al_inst_fg.unsqueeze(0)).item()
                    ue_inst_raw = edl_epistemic_scalar(al_inst_fg.unsqueeze(0)).item()
                    ua_inst, ue_inst = normalize_ua_ue_scalar(ua_inst_raw, ue_inst_raw, K=K_fg)
                    vac_inst = float(np.clip(K_fg / float(S_fg.item()), 0.0, 1.0))
                    conf_inst = float(p_inst_fg.max())

                    seg_ent_ins.append(ent)
                    seg_conf_ins.append(conf_inst)
                    seg_u_ins["total"].append(ent)
                    seg_u_ins["expected"].append(exp)
                    seg_u_ins["distributional"].append(dis)
                    seg_u_ins["edl_ale"].append(ua_inst)
                    seg_u_ins["edl_epi"].append(ue_inst)
                    seg_u_ins["vacuity"].append(vac_inst)

                    # correctness by MAJORITY semantic label inside the region
                    pred_region_labels = pred_major[region]
                    gt_region_labels   = gt_sem.detach().cpu().numpy()[region]
                    gt_fg = gt_region_labels[gt_region_labels > 0]

                    det_corr = (gt_fg.size > 0)

                    if gt_fg.size == 0:
                        seg_corr_ins.append(False)
                    else:
                        pred_majority = np.bincount(pred_region_labels).argmax()
                        gt_majority   = np.bincount(gt_fg).argmax()  # in 1..K-1
                        seg_corr_ins.append(bool(pred_majority == gt_majority))
                else:
                    continue

                # -------- CENTROID instance (if available) --------
                if have_cent:
                    # Gaussian-like map from centroid head
                    g_full = seed_prob_1[0]        # [H,W], assumed ≥0
                    region_vals = g_full[region]
                    if region_vals.size > 0:
                        peak_val = float(region_vals.max())
                        S_i      = float(region_vals.sum())
                    else:
                        peak_val = 0.0
                        S_i      = 0.0

                    # peak: 1 - normalized peak height
                    peak_norm = peak_val / max(G_MAX, 1e-8)
                    peak_norm = np.clip(peak_norm, 0.0, 1.0)
                    u_peak = 1.0 - peak_norm

                    # mass: |1 - S_i/100|
                    u_mass = abs(1.0 - S_i / 100.0)
                    u_mass = np.clip(u_mass, 0.0, 1.0)

                    # shift: tanh(distance between region CoM and region peak / σ)
                    coords_region = np.argwhere(region)
                    cm_rc = coords_region.mean(axis=0)  # (row, col)
                    peak_coords = coords_region[region_vals.argmax()] if region_vals.size > 0 else cm_rc
                    d_px   = float(np.linalg.norm(peak_coords - cm_rc))
                    u_shift = np.tanh(d_px / SIGMA_PX)

                    u_cent_inst = 0.3 * u_peak + 0.6 * u_mass + 0.0 * u_shift
                    u_cent_inst = float(np.clip(u_cent_inst, 0.0, 1.0))
                    for key in u_cent_ins:
                        if key == "peak":
                            u_cent_ins[key].append(u_peak)
                        elif key == "mass":
                            u_cent_ins[key].append(u_mass)
                        elif key == "shift":
                            u_cent_ins[key].append(u_shift)
                        elif key == "all":
                            u_cent_ins[key].append(u_cent_inst)
                    
                    cen_conf_ins.append(1.0 - u_cent_inst)
                    cen_corr_ins.append(det_corr)  # reuse seg correctness
            
        processed += B_eff

    # --------- stack helpers ----------
    def _stack(list_of_arrays):
        if len(list_of_arrays)==0: return np.array([])
        return np.concatenate(list_of_arrays, axis=0)

    # Pixel stacks
    seg_conf_pix = _stack(seg_conf_pix); seg_corr_pix = _stack(seg_corr_pix).astype(bool)
    for k in seg_u_pix: seg_u_pix[k] = _stack(seg_u_pix[k])

    # ### NEW: stacks for cell-union pixels
    seg_conf_pix_cell = _stack(seg_conf_pix_cell)
    seg_corr_pix_cell = _stack(seg_corr_pix_cell).astype(bool) if seg_conf_pix_cell.size else np.array([], bool)
    for k in seg_u_pix_cell:
        seg_u_pix_cell[k] = _stack(seg_u_pix_cell[k])

    # Instance stacks
    seg_conf_ins = np.asarray(seg_conf_ins, np.float32); seg_corr_ins = np.asarray(seg_corr_ins, bool)
    for k in seg_u_ins: seg_u_ins[k] = np.asarray(seg_u_ins[k], np.float32)
    seg_ent_ins = np.asarray(seg_ent_ins, np.float32)

    if len(cen_conf_ins):
        cen_conf_ins = np.asarray(cen_conf_ins, np.float32); cen_corr_ins = np.asarray(cen_corr_ins, bool)
        for k in u_cent_ins: u_cent_ins[k] = np.asarray(u_cent_ins[k], np.float32)
    else:
        cen_conf_ins = np.array([]); cen_corr_ins = np.array([], bool)

        # --------- 2D seg vs centroid uncertainty scatter plots (instance-level) ----------
    if seg_corr_ins.size and cen_corr_ins.size and len(u_cent_ins["all"]) > 0:
        try:
            plot_2d_seg_cent_uncert(
                seg_u_ins=seg_u_ins,
                cen_u_ins=u_cent_ins,
                seg_corr_ins=seg_corr_ins,
                cen_corr_ins=cen_corr_ins,
                out_root=out_root,
                max_points=int(cfg['evaluation'].get('scatter_max_points', 20000)),
            )
            print("Saved 2D seg-cent uncertainty scatter plots in:", out_root / "scatter_2d")

            plot_2d_detection_uncert(
                seg_u_ins = seg_u_ins,
                cen_u_ins = u_cent_ins,
                det_corr_ins = cen_corr_ins,
                out_root = out_root,
                max_points = int(cfg['evaluation'].get('scatter_max_points', 20000))
            )

            print()
        except Exception as e:
            print(f"(Warning) Failed to create 2D seg-cent scatter plots: {e}")


    

    # ------------- metrics + plots -------------
    def _block_multi(title_prefix, conf, corr, u_map: Dict[str, np.ndarray], subdir, *, n_classes: int):
        sub = out_root / subdir
        sub.mkdir(parents=True, exist_ok=True)
        results = {}

        # ECE (fixed + adaptive) with normalized confidence
        if conf is not None and conf.size > 0:
            conf01 = _normalize_confidence(conf, n_classes)
            ece = _ece_value(conf01, corr, bins)
            plot_ece(conf01, corr, bins, f"{title_prefix}", sub/"ece_fixed.png")
            mids, acc_q, conf_q, cnt_q, ace, mce, edges = compute_ece_adaptive(
                conf, corr.astype(np.float64), bins=bins, num_classes=n_classes)
            plot_reliability_adapt(acc_q, conf_q, cnt_q, ace,
                                   sub/"ece_adaptive.png",
                                   f"ACE - {title_prefix}",
                                   normalized=True, x=mids, x_mode="quantile",
                                   show_counts=False, show_gaps=True)
            results["ECE"] = ece; results["ACE"] = ace; results["MCE"] = mce

        # For each uncertainty proxy → UCE (fixed & adaptive) + diagnostics
        for name, u in u_map.items():
            if u.size == 0: continue
            key = name
            if name == "edl_ale":
                name = "Aleatoric"
            elif name == "edl_epi":
                name = "Epistemic"
            elif name == "vacuity":
                name = "Vacuity"
            uce = _uce_value(u, corr, bins)
            plot_uce(u, corr, bins, f"{title_prefix} — {name}", sub/f"uce_{key}.png")

            # Adaptive UCE (proxies already in [0,1] → do NOT rescale by K)
            mids_u, err_m, unc_m, adj_uce, max_uce, cnt_u, s, edges_u = compute_uce_adaptive(
                u, 1.0 - corr.astype(np.float64), bins=bins, n_classes=None, normalize=True
            )
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(mids_u, err_m, "o-", label="Mean error")
            ax.plot(mids_u, unc_m, "s--", label="Mean uncertainty")
            ax.plot([0,1],[0,1], "k:", lw=1)
            ax.set(xlabel="Uncertainty (quantile bins)", ylabel="Error", ylim=(0,1))
            ax.legend(loc="upper left"); plt.title(f"{title_prefix} — {name}\nAdj-UCE={adj_uce:.3f} | Max-UCE={max_uce:.3f}")
            plt.tight_layout(); plt.savefig(sub/f"uce_adaptive_{key}.png", dpi=200); plt.close()

            # Other metrics
            err = (~corr).astype(np.int32)
            auroc = _auroc(u, err)
            aurc  = risk_coverage_auc(u, corr)
            ks    = ks_2samp(u[corr], u[~corr]).statistic if corr.any() and (~corr).any() else np.nan

            plot_uncertainty_hist(u, corr, f"{title_prefix} — {name} - Uncertainty Distribution", sub/f"hist_{key}.png")
            plot_ecdf(u, corr, f"{title_prefix} — {name}", sub/f"ecdf_{key}.png")

            results.update({
                f"UCE[{key}]": uce,
                f"Adj_UCE[{key}]": adj_uce,
                f"Max_UCE[{key}]": max_uce,
                f"AUROC_error[{key}]": auroc,
                f"AURC[{key}]": aurc,
                f"KS[{key}]": ks,
                f"mean_{key}_correct": float(u[corr].mean()) if corr.any() else np.nan,
                f"mean_{key}_wrong":   float(u[~corr].mean()) if (~corr).any() else np.nan,
            })
        return results

    def _block_combined(title_prefix, corr, u_comb, subdir):
        sub = out_root / subdir
        sub.mkdir(parents=True, exist_ok=True)
        results = {}
        uce = _uce_value(u_comb, corr, bins)
        plot_uce(u_comb, corr, bins, f"{title_prefix} — mean(ent_seg, ent_cent)", sub/"uce.png")
        auroc = _auroc(u_comb, (~corr).astype(np.int32))
        aurc  = risk_coverage_auc(u_comb, corr)
        ks    = ks_2samp(u_comb[corr], u_comb[~corr]).statistic if corr.any() and (~corr).any() else np.nan

        # Adaptive UCE for combined
        mids_u, err_m, unc_m, adj_uce, max_uce, cnt_u, s, edges_u = compute_uce_adaptive(
            u_comb, 1.0 - corr.astype(np.float64), bins=bins, n_classes=None, normalize=True
        )
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(mids_u, err_m, "o-", label="Mean error")
        ax.plot(mids_u, unc_m, "s--", label="Mean uncertainty")
        ax.plot([0,1],[0,1], "k:", lw=1)
        ax.set(xlabel="Uncertainty (quantile bins)", ylabel="Error", ylim=(0,1))
        ax.legend(loc="upper left"); plt.title(f"{title_prefix}\nAdj-UCE={adj_uce:.3f} | Max-UCE={max_uce:.3f}")
        plt.tight_layout(); plt.savefig(sub/"uce_adaptive.png", dpi=200); plt.close()

        results.update({
            "UCE": uce,
            "Adj_UCE_adaptive": adj_uce,
            "Max_UCE_adaptive": max_uce,
            "AUROC_error": auroc,
            "AURC": aurc,
            "KS": ks
        })
        return results

    results = {
        "segmentation_pixel":
            _block_multi("Ours w",
                        conf=seg_conf_pix, corr=seg_corr_pix,
                        u_map=seg_u_pix, subdir="seg_pix", n_classes=K_seg),
    }
    # ### NEW: cell-union pixel metrics block
    if seg_conf_pix_cell.size:
        results["segmentation_pixel_cell_union"] = _block_multi(
            "Segmentation (pixel | cell-union)",
            conf=seg_conf_pix_cell, corr=seg_corr_pix_cell,
            u_map=seg_u_pix_cell, subdir="seg_pix_cell", n_classes=K_seg
        )

    # Use K_fg here (not K_seg)
    results["segmentation_instance"] = _block_multi("Ours w",
                        conf=seg_conf_ins, corr=seg_corr_ins,
                        u_map=seg_u_ins, subdir="seg_ins", n_classes=K_fg)

    if len(cen_conf_ins):
        results["centroid_instance"] = _block_multi("Ours w",
                        conf=cen_conf_ins, corr=cen_corr_ins,
                        u_map=u_cent_ins, subdir="cen_ins", n_classes=2)
    
    # --------- COMBINED instance uncertainties (seg EDL + centroid peak/mass/shift) ---------
    # if u_cent_ins["all"].size == seg_u_ins["edl_ale"].size:
    #     comb_results = {}
    #     for k in u_cent_ins:
    #         u_cent_arr = u_cent_ins[k].astype(np.float32)

    #         def mix_thr(us, uc, t=0.3, g=1.0):
    #             mix = us.copy()
    #             idx = uc > t
    #             mix[idx] = us[idx] + (1.0 - us[idx]) * (((uc[idx] - t) / (1.0 - t)) ** g)
    #             return mix

    #         def mix_mean(us, uc):
    #             return 0.5 * (us + uc)

    #         for key in ["edl_ale", "edl_epi", "vacuity"]:
    #             u_seg_arr = seg_u_ins[key].astype(np.float32)
    #             # safety: align length
    #             n = min(u_seg_arr.size, u_cent_arr.size, seg_corr_ins.size)
    #             u_seg_use = u_seg_arr[:n]
    #             u_cent_use = u_cent_arr[:n]
    #             corr_use = seg_corr_ins[:n]

    #             u_thr  = mix_thr(u_seg_use, u_cent_use, t=0.3, g=1.0)
    #             u_mean = mix_mean(u_seg_use, u_cent_use)

    #             name_thr  = f"{key}_{k}_thr"
    #             name_mean = f"{key}_{k}_mean"

    #             comb_results[name_thr] = _block_combined(
    #                 f"Combined — {key} + {k} (thr)",
    #                 corr_use, u_thr, subdir=f"comb_ins_{name_thr}"
    #             )
    #             comb_results[name_mean] = _block_combined(
    #                 f"Combined — {key} + {k} (mean)",
    #                 corr_use, u_mean, subdir=f"comb_ins_{name_mean}"
    #             )

    #             comb_results[name_thr] = _block_multi(
    #                 f"Combined — {key} + {k} (thr)",
    #                 conf=1-u_thr, corr=corr_use,
    #                 u_map={name_thr: u_thr},
    #                 subdir=f"comb_ins_{name_thr}", n_classes=1
    #             )

    #             comb_results[name_mean] = _block_multi(
    #                 f"Combined — {key} + {k} (mean)",
    #                 conf=1-u_mean, corr=corr_use,
    #                 u_map={name_mean: u_mean},
    #                 subdir=f"comb_ins_{name_mean}", n_classes=1
    #             )

    #     results["combined_instance"] = comb_results
    # else:
    #     print("(Note) Combined instance EDL+centroid metrics skipped: no centroid head / u_cent_ins empty.")


    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root/"metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved overlays in:", out_root / "viz")
    print("Saved metrics in:", out_root / "metrics.json")
    if not len(cen_conf_ins):
        print("(Note) Centroid + combined metrics were skipped because the checkpoint had no centroid head.)")

# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Uncertainty evaluation (DualUNet EDL)")
    parser.add_argument('--config-file', type=str, required=True, help='Path to config file.')
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Override options like key1=value1 key2=value2")
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    # simple overrides with basic typing
    if args.opts:
        for opt in args.opts:
            k, v = opt.split('=')
            if v.lower() in ('true', 'false'):
                v = (v.lower() == 'true')
            else:
                try:
                    if '.' in v: v = float(v)
                    else: v = int(v)
                except ValueError:
                    pass
            d = cfg
            keys = k.split('.')
            for kk in keys[:-1]:
                d = d[kk]
            d[keys[-1]] = v

    run_uncertainty(cfg)

if __name__ == "__main__":
    main()
