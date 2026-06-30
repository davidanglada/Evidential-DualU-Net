"""Dirichlet pixel- and instance-level uncertainty utilities."""

from typing import Literal
import math
import torch


def dirichlet_probabilities(alpha: torch.Tensor, class_dim: int = 1) -> torch.Tensor:
    """Return the predictive categorical mean ``alpha / sum(alpha)``."""
    if alpha.ndim < 1 or torch.any(alpha <= 0):
        raise ValueError("alpha must contain positive Dirichlet parameters")
    return alpha / alpha.sum(dim=class_dim, keepdim=True).clamp_min(1e-12)


def dirichlet_uncertainty(alpha: torch.Tensor, class_dim: int = 1) -> dict[str, torch.Tensor]:
    """Compute normalized entropy, aleatoric, epistemic, MI, and vacuity maps."""
    alpha = alpha.movedim(class_dim, -1)
    classes = alpha.shape[-1]
    if classes < 2 or torch.any(alpha <= 0):
        raise ValueError("alpha must have at least two positive class parameters")
    strength = alpha.sum(dim=-1)
    probs = alpha / strength.unsqueeze(-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(classes)
    expected_entropy = -(
        probs * (torch.digamma(alpha + 1.0) - torch.digamma(strength.unsqueeze(-1) + 1.0))
    ).sum(-1) / math.log(classes)
    aleatoric = (alpha * (strength.unsqueeze(-1) - alpha)).sum(-1) / (strength * (strength + 1.0))
    epistemic = (probs * (1.0 - probs) / (strength.unsqueeze(-1) + 1.0)).sum(-1)
    return {
        "entropy": entropy.clamp(0, 1),
        "aleatoric": (aleatoric / ((classes - 1.0) / classes)).clamp(0, 1),
        "epistemic": (epistemic / ((classes - 1.0) / (classes * (classes + 1.0)))).clamp(0, 1),
        "mutual_information": (entropy - expected_entropy).clamp(0, 1),
        "vacuity": (classes / strength).clamp(0, 1),
        "strength": strength,
    }


def pool_instance_uncertainty(
    alpha: torch.Tensor,
    instances: torch.Tensor,
    reduction: Literal["mean", "median", "sum"] = "mean",
    background_index: int | None = 0,
) -> dict[int, dict[str, float]]:
    """Compute size-invariant, instance-level evidential uncertainty.

    ``alpha`` is ``[K,H,W]`` and ``instances`` is ``[H,W]``. Following the
    paper, the default operation averages Dirichlet parameters within each
    nucleus and excludes the background class before computing uncertainty.
    ``median`` and ``sum`` are provided for reproducing the pooling ablation.
    """
    if alpha.ndim != 3 or instances.shape != alpha.shape[1:]:
        raise ValueError("expected alpha [K,H,W] and matching instances [H,W]")
    if reduction not in {"mean", "median", "sum"}:
        raise ValueError("reduction must be 'mean', 'median', or 'sum'")
    class_ids = torch.arange(alpha.shape[0], device=alpha.device)
    if background_index is not None:
        if not 0 <= background_index < alpha.shape[0]:
            raise ValueError("background_index is outside the class dimension")
        keep = class_ids != background_index
        alpha = alpha[keep]
        class_ids = class_ids[keep]
    if alpha.shape[0] < 2:
        raise ValueError("at least two foreground classes are required")
    result: dict[int, dict[str, float]] = {}
    for label in torch.unique(instances).tolist():
        label = int(label)
        if label <= 0:
            continue
        pixels = alpha[:, instances == label]
        if reduction == "mean":
            pooled = pixels.mean(1)
        elif reduction == "median":
            pooled = pixels.median(1).values
        else:
            pooled = pixels.sum(1)
        unc = dirichlet_uncertainty(pooled.unsqueeze(0), class_dim=1)
        probs = dirichlet_probabilities(pooled, class_dim=0)
        result[label] = {
            "class_id": int(class_ids[probs.argmax()].item()),
            "confidence": float(probs.max().item()),
            **{k: float(v.item()) for k, v in unc.items()},
        }
    return result
