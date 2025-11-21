from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma


class Denoiser:
    """Apply basic denoising filters to 3D volumes.

    The methods here are intentionally minimal but practical enough
    for typical tomography stacks.
    """

    def nlm(
        self,
        volume: np.ndarray,
        patch_size: int = 3,
        patch_distance: int = 5,
        h_factor: float = 1.0,
        fast_mode: bool = True,
    ) -> np.ndarray:
        """Non-local means denoising.

        Parameters
        ----------
        volume:
            Input volume of shape (Z, Y, X).
        patch_size:
            Size of patches used for denoising.
        patch_distance:
            Max distance in pixels to search for similar patches.
        h_factor:
            Multiplier on the estimated noise standard deviation.
        fast_mode:
            Use faster approximate NLM if True.

        Returns
        -------
        np.ndarray
            Denoised volume with the same shape as input.
        """
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        # skimage expects (Z, Y, X) for 3D as well
        sigma_est = np.mean(estimate_sigma(volume, channel_axis=None))
        h = h_factor * sigma_est

        denoised = denoise_nl_means(
            volume,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=fast_mode,
            channel_axis=None,
            preserve_range=True,
        )
        return denoised.astype(volume.dtype)

    def gaussian(self, volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply 3D Gaussian filtering.

        Parameters
        ----------
        volume:
            Input volume of shape (Z, Y, X).
        sigma:
            Standard deviation of the Gaussian kernel (in pixels).
        """
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        blurred = gaussian_filter(volume, sigma=sigma)
        return blurred.astype(volume.dtype)