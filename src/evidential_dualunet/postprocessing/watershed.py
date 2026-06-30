"""Watershed reconstruction from semantic and centroid predictions."""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def reconstruct_instances(
    foreground: np.ndarray,
    centroid: np.ndarray | None = None,
    foreground_threshold: float = 0.5,
    centroid_threshold: float = 0.15,
    min_distance: int = 3,
) -> np.ndarray:
    """Return an integer instance mask from 2-D foreground/centroid maps."""
    if foreground.ndim != 2:
        raise ValueError("foreground must be a 2-D array")
    mask = np.asarray(foreground) >= foreground_threshold
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(mask)
    score = distance if centroid is None else np.asarray(centroid)
    if score.shape != mask.shape:
        raise ValueError("centroid and foreground shapes must match")
    coordinates = peak_local_max(score, min_distance=min_distance, threshold_abs=centroid_threshold, labels=mask)
    markers = np.zeros(mask.shape, dtype=np.int32)
    if len(coordinates):
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    else:
        markers, _ = ndi.label(mask)
    return watershed(-distance, markers, mask=mask).astype(np.int32)

