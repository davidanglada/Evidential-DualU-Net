"""Geometric uncertainty cues for centroid Gaussian maps."""

import math
import numpy as np
import torch


def centroid_uncertainty(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize uncertainty fields emitted by Beta or NIG centroid heads."""
    if "sigma2_ale" in output:
        return {"aleatoric": output["sigma2_ale"], "epistemic": output["sigma2_epi"], "evidence": output["S"]}
    if "alpha" in output:
        alpha = output["alpha"]
        strength = alpha.sum(1, keepdim=True)
        return {"vacuity": alpha.shape[1] / strength, "evidence": strength}
    prediction = output.get("y_hat", output.get("p_cent"))
    if prediction is None:
        raise ValueError("centroid output has no recognized prediction or evidential fields")
    return {"peak_ambiguity": 1.0 - prediction.clamp(0, 1)}


def geometric_centroid_uncertainty(
    centroid_map: np.ndarray,
    instances: np.ndarray,
    sigma: float = 5.0,
    peak_weight: float = 0.3,
    mass_weight: float = 0.6,
) -> dict[int, dict[str, float]]:
    """Compute the paper's peak, mass-ratio, and combined centroid scores.

    The expected mass of the isotropic Gaussian target is ``2*pi*sigma**2``.
    Inputs must use the same intensity scale as the Gaussian target; undo any
    training-time scale factor before calling this function.
    """
    centroid_map = np.asarray(centroid_map, dtype=np.float64)
    instances = np.asarray(instances)
    if centroid_map.ndim != 2 or instances.shape != centroid_map.shape:
        raise ValueError("expected matching 2-D centroid and instance maps")
    expected_mass = 2.0 * math.pi * sigma**2
    result: dict[int, dict[str, float]] = {}
    for label in np.unique(instances):
        label = int(label)
        if label <= 0:
            continue
        values = centroid_map[instances == label]
        peak = 1.0 - float(values.max())
        mass = abs(float(values.sum()) - expected_mass) / expected_mass
        result[label] = {
            "peak": peak,
            "mass": mass,
            "combined": peak_weight * peak + mass_weight * mass,
        }
    return result
