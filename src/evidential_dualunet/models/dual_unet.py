"""Thin public wrapper around the original checkpoint-compatible architecture."""

from typing import Any


def build_model(config: dict[str, Any], *, monte_carlo_dropout: bool = False):
    """Build DualU-Net from a repository configuration dictionary.

    The import is lazy so uncertainty-only use does not require model extras.
    """
    from dual_unet.models import build_model as _legacy_build_model

    return _legacy_build_model(config, mcd=monte_carlo_dropout)

