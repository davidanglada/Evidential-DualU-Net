#!/usr/bin/env python3
"""Render uncertainty maps saved by scripts/infer.py."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize uncertainty maps from an inference .npz file.")
    parser.add_argument("--image", required=True, help="Original RGB image")
    parser.add_argument("--predictions", required=True, help="Prediction .npz from infer.py")
    parser.add_argument("--output", required=True, help="Destination PNG")
    args = parser.parse_args()
    for value in (args.image, args.predictions):
        if not Path(value).is_file(): parser.error(f"File not found: {value}")
    from evidential_dualunet.visualization import save_uncertainty_figure
    data = np.load(args.predictions)
    names = ["aleatoric", "epistemic", "vacuity", "entropy", "mutual_information"]
    maps = {name: np.squeeze(data[name]) for name in names if name in data}
    save_uncertainty_figure(np.asarray(Image.open(args.image).convert("RGB")), maps, args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__": main()

