#!/usr/bin/env python3
"""Single-image, single-pass inference CLI."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from _common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-pass Evidential DualU-Net inference on one RGB image.")
    parser.add_argument("--config", required=True, help="Model YAML")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth checkpoint")
    parser.add_argument("--image", required=True, help="Input RGB image")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for value, label in [(args.checkpoint, "Checkpoint"), (args.image, "Image")]:
        if not Path(value).is_file(): parser.error(f"{label} not found: {value}")
    cfg = load_config(args.config)
    from evidential_dualunet.models import build_model
    from evidential_dualunet.inference import load_checkpoint, predict_tensor
    mean = torch.tensor(cfg.get("transforms", {}).get("normalize", {}).get("mean", [0.485, 0.456, 0.406]))[:, None, None]
    std = torch.tensor(cfg.get("transforms", {}).get("normalize", {}).get("std", [0.229, 0.224, 0.225]))[:, None, None]
    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1) - mean) / std
    model = load_checkpoint(build_model(cfg), args.checkpoint)
    output = predict_tensor(model, tensor.unsqueeze(0), args.device)
    arrays = {"probabilities": output["seg"]["p_hat"][0].cpu().numpy(), "alpha": output["seg"]["alpha"][0].cpu().numpy()}
    arrays.update({name: value[0].cpu().numpy() for name, value in output["uncertainty"].items()})
    if "cent" in output:
        cent = output["cent"]
        if torch.is_tensor(cent): arrays["centroid"] = cent[0].cpu().numpy()
        elif "y_hat" in cent: arrays["centroid"] = cent["y_hat"][0].cpu().numpy()
        elif "p_cent" in cent: arrays["centroid"] = cent["p_cent"][0].cpu().numpy()
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"Saved predictions to {destination}")


if __name__ == "__main__": main()

