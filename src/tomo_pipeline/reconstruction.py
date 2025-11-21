from __future__ import annotations

import numpy as np


class VolumeReconstructor:
    """High-level helpers to post-process volumes.

    This is intentionally lightweight; think of it as a place to
    centralize common post-processing steps:
    - intensity normalization
    - cropping margins
    - optional hooks for further transforms
    """

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensities to [0, 1] range."""
        vmin = float(volume.min())
        vmax = float(volume.max())
        if vmax <= vmin:
            return np.zeros_like(volume, dtype=np.float32)

        norm = (volume - vmin) / (vmax - vmin)
        return norm.astype(np.float32)

    def crop_volume(self, volume: np.ndarray, margin: int = 0) -> np.ndarray:
        """Crop away a constant margin along Y / X (and optionally Z)."""
        if margin <= 0:
            return volume

        z, y, x = volume.shape
        z_slice = slice(margin, max(z - margin, margin + 1))
        y_slice = slice(margin, max(y - margin, margin + 1))
        x_slice = slice(margin, max(x - margin, margin + 1))
        cropped = volume[z_slice, y_slice, x_slice]
        return cropped

    def build_volume(self, stack: np.ndarray, margin: int = 0) -> np.ndarray:
        """Combine pre-processing + cropping + normalization.

        Parameters
        ----------
        stack:
            Input volume/stack of shape (Z, Y, X).
        margin:
            Optional margin to crop from each side.

        Returns
        -------
        np.ndarray
            Final processed volume in [0, 1].
        """
        pre = self.preprocess(stack)
        out = self.crop_volume(pre, margin=margin)
        return out