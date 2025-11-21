from __future__ import annotations

from typing import Literal

import numpy as np
from skimage.filters import threshold_otsu

SegMethod = Literal["otsu", "global"]


class Segmenter:
    """Pluggable segmentation backend (threshold-based for now).

    This starts simple (Otsu / global threshold) but can be extended
    to use more advanced ML-based segmentation later.
    """

    def __init__(self, method: SegMethod = "otsu", model_path: str | None = None):
        self.method = method
        self.model_path = model_path
        # model_path is a placeholder for future ML backends; unused for now.

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Return a binary mask.

        Parameters
        ----------
        volume:
            Input volume of shape (Z, Y, X).

        Returns
        -------
        np.ndarray
            Boolean mask of the same shape as `volume`.
        """
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        if self.method == "otsu":
            # Flatten volume for a single global threshold
            thresh = threshold_otsu(volume)
            mask = volume >= thresh
            return mask

        if self.method == "global":
            # Simple fixed threshold at mid-range
            vmin, vmax = float(volume.min()), float(volume.max())
            thresh = (vmin + vmax) / 2.0
            mask = volume >= thresh
            return mask

        raise ValueError(f"Unknown segmentation method: {self.method}")