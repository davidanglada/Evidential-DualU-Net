"""Small dependency-light metrics for programmatic evaluation."""

import numpy as np


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Compute binary Dice similarity."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    return float((2 * np.logical_and(prediction, target).sum() + eps) / (prediction.sum() + target.sum() + eps))

