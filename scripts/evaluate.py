#!/usr/bin/env python3
"""Public evaluation entry point."""

import argparse
import runpy
import sys
from pathlib import Path
from _common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Evidential DualU-Net checkpoint.")
    parser.add_argument("--config", required=True, help="Evaluation YAML")
    parser.add_argument("--uncertainty", action="store_true", help="Also run uncertainty-aware evaluation")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override nested config values")
    args = parser.parse_args(); load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    legacy = root / ("eval_unc.py" if args.uncertainty else "eval.py")
    sys.argv = [str(legacy), "--config-file", args.config]
    if args.set: sys.argv += ["--opts", *args.set]
    runpy.run_path(str(legacy), run_name="__main__")


if __name__ == "__main__": main()

