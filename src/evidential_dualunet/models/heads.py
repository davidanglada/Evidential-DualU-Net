"""Public aliases for the evidential heads used by released checkpoints."""

from dual_unet.models.base.heads import (
    EvidentialCentroidHeadBeta,
    EvidentialCentroidHeadNIG,
    EvidentialSegmentationHead,
)

__all__ = [
    "EvidentialSegmentationHead",
    "EvidentialCentroidHeadBeta",
    "EvidentialCentroidHeadNIG",
]

