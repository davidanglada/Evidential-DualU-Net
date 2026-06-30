#!/usr/bin/env python3
"""Public training entry point; delegates to the checkpoint-compatible trainer."""

import argparse
import runpy
import sys
from pathlib import Path
from _common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Evidential DualU-Net from a YAML configuration.")
    parser.add_argument("--config", required=True, help="Training YAML, e.g. configs/train_pannuke.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override nested config values")
    args = parser.parse_args()
    load_config(args.config)
    legacy = Path(__file__).resolve().parents[1] / "train.py"
    sys.argv = [str(legacy), "--config-file", args.config]
    if args.set: sys.argv += ["--opts", *args.set]
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__": main()

