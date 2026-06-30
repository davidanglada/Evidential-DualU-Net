#!/usr/bin/env python3
# engine_unet_edl.py
#
# Engine for training/evaluating an Evidential UNet (segmentation-only).

import math
import sys
import time
from typing import Iterable, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import dual_unet.utils.misc as utils
# Make sure UNetEvaluationMetric is importable from your eval module
# If you saved it as unet_eval_watershed_circles.py, adjust the path accordingly:
from dual_unet.eval import UNetEvaluationMetric


# ------------------------------------------------------------------
#  TRAIN  ONE  EPOCH  (UNet + EDL seg head)
# ------------------------------------------------------------------
def train_one_epoch(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
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

        # ── FORWARD ────────────────────────────────────────────────
        # Expected model output (EDL UNet seg head):
        # out = {"alpha": [B,K,H,W], "p_hat": [B,K,H,W], "S": [B,1,H,W]}
        out = model(samples)

        # ── LOSS (EDL segmentation-only; criterion returns tuple) ──
        # Expected return signature:
        # loss_all, loss_seg, loss_dice, loss_kl, stats
        # where stats includes: mean_S_seg, mean_max_p_seg, misclassified_frac, kl_weight, ...
        loss_all, loss_seg, loss_dice, loss_kl, stats = criterion(out, targets, epoch=epoch)

        # ── BACKWARD ───────────────────────────────────────────────
        loss_all.backward()
        if max_norm and max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # ── LOGGING ────────────────────────────────────────────────
        metric_logger.update(
            loss=loss_all.item(),
            loss_seg=loss_seg.item() if torch.is_tensor(loss_seg) else float(loss_seg),
            loss_dice=loss_dice.item() if torch.is_tensor(loss_dice) else float(loss_dice),
            loss_kl=loss_kl.item() if torch.is_tensor(loss_kl) else float(loss_kl),
            mean_S_seg=float(stats.get("mean_S_seg", 0.0)),
            mean_max_p_seg=float(stats.get("mean_max_p_seg", 0.0)),
            misclassified_frac=float(stats.get("misclassified_frac", 0.0)),
            kl_weight=float(stats.get("kl_weight", 0.0)),
            lr=optimizer.param_groups[0]["lr"],
            grad_norm=grad_total_norm,
        )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ------------------------------------------------------------------
#  EVALUATION   (UNet + EDL seg head)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    h: float = 0.1,              # h-maxima for watershed markers
    eval_mode: bool = True,      # if False, skip instance-level metrics even if masks exist
) -> dict:

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # Watershed-based evaluator on semantic map
    metrics = {
        "seg": UNetEvaluationMetric(
            num_classes=data_loader.dataset.num_classes,
            class_names=getattr(data_loader.dataset, "class_names", None),
            eval_mode=eval_mode,
            h=h,
            dataset_tag=getattr(data_loader.dataset, "name", "circles"),
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, "Val:"):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in t.items()}
            for t in targets
        ]

        # ── forward ────────────────────────────────────────────────
        out = model(samples)

        # --------- loss (epoch=0 ⇒ full KL in many schedules) -----
        loss_all, loss_seg, loss_dice, loss_kl, stats = criterion(out, targets, epoch=0)
        metric_logger.update(
            loss=loss_all.item(),
            loss_seg=loss_seg.item() if torch.is_tensor(loss_seg) else float(loss_seg),
            loss_dice=loss_dice.item() if torch.is_tensor(loss_dice) else float(loss_dice),
            loss_kl=loss_kl.item() if torch.is_tensor(loss_kl) else float(loss_kl),
            mean_S_seg=float(stats.get("mean_S_seg", 0.0)),
            mean_max_p_seg=float(stats.get("mean_max_p_seg", 0.0)),
            misclassified_frac=float(stats.get("misclassified_frac", 0.0)),
            kl_weight=float(stats.get("kl_weight", 0.0)),
        )

        # --------- pack predictions for evaluator -----------------
        # Prefer evidential predictive mean
        p_seg = out['seg']['p_hat']  # [B,K,H,W]
        preds = [
            {
                "p_hat": p_seg[i],     # evaluator will use this; falls back to "segmentation_mask" if absent
                "image": samples[i],
            }
            for i in range(samples.size(0))
        ]
        metrics["seg"].update(preds, targets)

    metric_logger.synchronize_between_processes()
    losses = {k: m.global_avg for k, m in metric_logger.meters.items()}
    metric_vals_nested = {k: metrics[k].compute() for k in metrics}
    metric_vals = {}
    for head, vals in metric_vals_nested.items():
        for k, v in vals.items():
            metric_vals[f"{head}/{k}"] = v

    return {**losses, **metric_vals}

# ------------------------------------------------------------------
#  EVALUATION  (TEST-TIME, with visual dumps via evaluator)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_test(
    cfg: dict,
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    h: float = 0.1,              # h-maxima for watershed markers
    eval_mode: bool = True,      # if False -> skip instance metrics even if masks exist
    output_suffix: str = "",
) -> dict:

    model.eval()
    criterion.eval()  # kept for symmetry; not needed for metrics

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    evaluator = {
        "seg": UNetEvaluationMetric(
                num_classes=data_loader.dataset.num_classes,
                class_names=getattr(data_loader.dataset, "class_names", None),
                eval_mode=eval_mode,
                h=h,
                output_suffix=output_suffix if output_suffix else None,
                dataset_tag=getattr(data_loader.dataset, "name", "circles"),
        )
    }

    for samples, targets in metric_logger.log_every(data_loader, 10, "Val:"):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in t.items()}
            for t in targets
        ]


        # ── forward ────────────────────────────────────────────────
        out = model(samples)  # expects {"alpha","p_hat","S"}

        p_seg = out['seg']['p_hat']  # [B,K,H,W]
        preds = [
            {
                "p_hat": p_seg[i],     # evaluator will use this; falls back to "segmentation_mask" if absent
                "image": samples[i],
            }
            for i in range(samples.size(0))
        ]
        evaluator["seg"].update(preds, targets)

    metric_logger.synchronize_between_processes()
    metric_vals_nested = {k: evaluator[k].compute() for k in evaluator}
    metric_vals = {}
    for head, vals in metric_vals_nested.items():
        for k, v in vals.items():
            metric_vals[f"{head}/{k}"] = v

    return {**metric_vals}
