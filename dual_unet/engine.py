import math
import sys
import time
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import dual_unet.utils.misc as utils
from dual_unet.eval import (
    MultiTaskEvaluationMetric,
    MultiTaskEvaluationMetric_all,
    MultiTaskEvaluationMetric_op,
    MultiTaskEvaluationMetric_unc,
)

# ------------------------------
# helpers
# ------------------------------
def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


# ------------------------------------------------------------------
#  TRAIN  ONE  EPOCH  (EDL + NIG heads)
# ------------------------------------------------------------------
def train_one_epoch(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    thresholds: List[float] = (0.5,),
    max_pair_distance: float = 12.0,
    max_norm: float = 0,
    th: float = 0.15,
) -> dict:

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(1, "{value:.6f}"))
    header, print_freq = f"Epoch: [{epoch}]", 1

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()

        # ── FORWARD ──────────────────────────────────────────────────
        out = model(samples)

        # loss_all, loss_seg_data, loss_cent_data, loss_dice,
        # loss_kl_seg, loss_kl_cent, loss_kl_total, stats
        (
            loss_all,
            loss_seg,
            loss_cent,
            loss_dice,
            loss_kl_seg,
            loss_kl_total,
            stats,
        ) = criterion(out, targets, epoch=epoch)

        # ── BACKWARD ─────────────────────────────────────────────────
        loss_all.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # ── LOGGING ──────────────────────────────────────────────────
        metric_logger.update(
            loss=_to_float(loss_all),
            loss_seg=_to_float(loss_seg),
            loss_cent=_to_float(loss_cent),
            loss_dice=_to_float(loss_dice),

            loss_kl_total=_to_float(loss_kl_total),
            loss_kl=_to_float(loss_kl_total),

            mean_S_seg=float(stats.get("mean_S_seg", 0.0)),
            mean_max_p_seg=float(stats.get("mean_max_p_seg", 0.0)),
            misclassified_frac=float(stats.get("misclassified_frac", 0.0)),

            kl_weight=float(stats.get("kl_weight", 0.0)),
            kl_weight_seg=float(stats.get("kl_weight_seg", stats.get("kl_weight", 0.0))),

            lr=optimizer.param_groups[0]["lr"] if "optimizer" in locals()
               else metric_logger.meters.get("lr", 0.0),
            grad_norm=grad_total_norm if "grad_total_norm" in locals() else 0.0
        )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ------------------------------------------------------------------
#  EVALUATION   (EDL + NIG heads)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    max_pair_distance: float = 12.0,
    th: float = 0.15,
) -> dict:

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metrics = {
        "f": MultiTaskEvaluationMetric(
            num_classes=data_loader.dataset.num_classes,
            max_pair_distance=max_pair_distance,
            class_names=getattr(data_loader.dataset, "class_names", None),
            train=True,
            th=th,
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, "Val:"):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in t.items()}
            for t in targets
        ]

        out = model(samples)

        (
            loss_all,
            loss_seg,
            loss_cent,
            loss_dice,
            loss_kl_seg,
            loss_kl_total,
            stats,
        ) = criterion(out, targets, epoch=0)

        metric_logger.update(
            loss=_to_float(loss_all),
            loss_seg=_to_float(loss_seg),
            loss_cent=_to_float(loss_cent),
            loss_dice=_to_float(loss_dice),

            loss_kl_seg=_to_float(loss_kl_seg),
            loss_kl_total=_to_float(loss_kl_total),
            loss_kl=_to_float(loss_kl_total),

            mean_S_seg=float(stats.get("mean_S_seg", 0.0)),
            mean_max_p_seg=float(stats.get("mean_max_p_seg", 0.0)),
            misclassified_frac=float(stats.get("misclassified_frac", 0.0)),
            mean_S_cent=float(stats.get("mean_S_cent", 0.0)),

            kl_weight=float(stats.get("kl_weight", 0.0)),
            kl_weight_seg=float(stats.get("kl_weight_seg", stats.get("kl_weight", 0.0))),
            kl_weight_cent=float(stats.get("kl_weight_cent", stats.get("kl_weight", 0.0))),
        )

        # ------------------ NEW FOR NIG HEAD ------------------
        p_seg = out["seg"]["p_hat"]          # [B,K,H,W]
        p_cent = out["cent"]      # [B,1,H,W]  <-- NIG prediction

        preds = [
            {
                "segmentation_mask": p_seg[i],
                "centroid_prob": p_cent[i],   # evaluator still expects this key
                "image": samples[i],
            }
            for i in range(samples.size(0))
        ]
        for k in metrics:
            metrics[k].update(preds, targets)

    metric_logger.synchronize_between_processes()
    losses = {k: m.global_avg for k, m in metric_logger.meters.items()}
    metric_vals = {k: metrics[k].compute() for k in metrics}
    return {**losses, **metric_vals}


# ------------------------------------------------------------------
#  EVALUATION (TEST-TIME)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_test(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = (0.5,),
    max_pair_distance: float = 12.0,
    output_sufix: str = "",
    train: bool = False,
    th: float = 0.15,
) -> dict:

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    metrics = {
        "f": MultiTaskEvaluationMetric_all(
            num_classes=data_loader.dataset.num_classes,
            max_pair_distance=max_pair_distance,
            thresholds=thresholds,
            class_names=getattr(data_loader.dataset, "class_names", None),
            dataset=cfg["dataset"]["test"]["name"],
            train=False,
            th=th,
            output_sufix=output_sufix,
        )
    }

    GT_MAX = 0.00  # For debugging centroid predictions

    done = False

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in t.items()}
            for t in targets
        ]

        out = model(samples)

        ##### FOR DEBUGGING: print out["cent"]["x"] as a heatmap #####
        if not done:
            import matplotlib.pyplot as plt
            import numpy as np
            cent_x = out["x_cent"][0][0].cpu().numpy()
            plt.imshow(cent_x, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.show()
            plt.savefig('cent_x_debug.png')
            done = True
        ###############################################################

        p_seg = out["seg"]["p_hat"]         # [B,K,H,W]
        p_cent = out["cent"]    # [B,1,H,W]

        preds = [
            {
                "segmentation_mask": p_seg[i],
                "centroid_prob": p_cent[i],
                "image": samples[i],
            }
            for i in range(samples.size(0))
        ]

        for k in metrics:
            metrics[k].update(preds, targets)

    metric_logger.synchronize_between_processes()
    return {k: metrics[k].compute() for k in metrics}


@torch.no_grad()
def evaluate_test_unc(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    thresholds: List[float] = (0.5,),
    max_pair_distance: float = 12.0,
    output_sufix: str = "",
    train: bool = False,
    th: float = 0.15,
) -> dict:

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # --- main multitask metric (seg + cent + uncertainties) ---
    metrics = {
        "f": MultiTaskEvaluationMetric_unc(
            num_classes=data_loader.dataset.num_classes,
            max_pair_distance=max_pair_distance,
            thresholds=thresholds,
            class_names=getattr(data_loader.dataset, "class_names", None),
            dataset=cfg["dataset"]["test"]["name"],
            train=train,      # usually False for test
            th=th,
            output_sufix=output_sufix,
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in t.items()
            }
            for t in targets
        ]

        # --- forward ---
        out = model(samples)

        # Segmentation head
        # p_seg: [B,K,H,W] (mean probabilities)
        # alpha_seg: [B,K,H,W] (Dirichlet parameters)
        p_seg     = out["seg"]["p_hat"]
        alpha_seg = out["seg"]["alpha"]

        # Centroid head: Gaussian-like map [B,1,H,W]
        # (adapt this key if your model uses a different one)
        p_cent = out["cent"]   # already [B,1,H,W] in your previous code

        # --- build per-image prediction dicts for the metric ---
        preds = []
        for i in range(samples.size(0)):
            preds.append({
                "segmentation_mask": p_seg[i],      # [K,H,W]
                "alpha_seg":         alpha_seg[i],  # [K,H,W]
                "centroid_prob":     p_cent[i],     # [1,H,W] (Gaussian map)
                "image":             samples[i],
            })

        # --- update metrics ---
        for k in metrics:
            metrics[k].update(preds, targets)

    metric_logger.synchronize_between_processes()

    # Each metric object returns its dict (with raw + filtered metrics)
    return {k: metrics[k].compute() for k in metrics}
