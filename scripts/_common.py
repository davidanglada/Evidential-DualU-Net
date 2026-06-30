"""Shared command-line helpers (not a public API)."""

import json
from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_metrics(metrics: dict, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, default=float) + "\n")

