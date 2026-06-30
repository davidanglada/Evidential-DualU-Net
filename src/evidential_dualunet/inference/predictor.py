"""Single-pass model inference and checkpoint loading."""

from pathlib import Path
from typing import Any
import torch
from evidential_dualunet.uncertainty import dirichlet_uncertainty


def load_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = True) -> torch.nn.Module:
    """Load a plain or ``{'model': state_dict}`` checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=strict)
    return model


@torch.inference_mode()
def predict_tensor(model: torch.nn.Module, image: torch.Tensor, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Run one forward pass and append uncertainty maps to model outputs."""
    model = model.to(device).eval()
    output = model(image.to(device))
    if not isinstance(output, dict) or "seg" not in output or "alpha" not in output["seg"]:
        raise RuntimeError("Expected model output['seg']['alpha']; checkpoint/model may be incompatible")
    output["uncertainty"] = dirichlet_uncertainty(output["seg"]["alpha"], class_dim=1)
    return output

