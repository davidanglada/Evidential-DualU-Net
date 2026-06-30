"""Run setup and reproducibility helpers."""

import json
import os
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def prepare_run(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Create a run directory and save its resolved configuration as JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.resolved.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    return path

