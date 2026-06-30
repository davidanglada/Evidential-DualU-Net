"""Uncertainty-map visualization."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_uncertainty_figure(image: np.ndarray, maps: dict[str, np.ndarray], output: str | Path) -> None:
    """Save an image and named uncertainty maps as one compact PNG figure."""
    count = len(maps) + 1
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), squeeze=False)
    axes[0, 0].imshow(image); axes[0, 0].set_title("image")
    for axis, (name, values) in zip(axes[0, 1:], maps.items()):
        plot = axis.imshow(values, cmap="magma", vmin=0, vmax=1)
        axis.set_title(name); fig.colorbar(plot, ax=axis, fraction=0.046)
    for axis in axes.flat: axis.axis("off")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output, dpi=150, bbox_inches="tight"); plt.close(fig)

