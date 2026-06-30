"""Checkpoint-compatible training losses."""

from dual_unet.models.losses import DualLoss_Evidential, SegLoss_Evidential

__all__ = ["DualLoss_Evidential", "SegLoss_Evidential"]

