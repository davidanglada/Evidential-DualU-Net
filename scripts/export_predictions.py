#!/usr/bin/env python3
"""Export compressed prediction arrays to portable per-map .npy files."""

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export each field of a prediction .npz archive.")
    parser.add_argument("--input", required=True, help="Input .npz archive")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy fields")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file(): parser.error(f"Prediction archive not found: {source}")
    destination = Path(args.output_dir); destination.mkdir(parents=True, exist_ok=True)
    with np.load(source) as data:
        for name in data.files: np.save(destination / f"{name}.npy", data[name])
    print(f"Exported fields to {destination}")


if __name__ == "__main__": main()

