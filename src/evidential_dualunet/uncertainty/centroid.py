"""Geometric and evidential uncertainty cues for centroid maps."""

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

