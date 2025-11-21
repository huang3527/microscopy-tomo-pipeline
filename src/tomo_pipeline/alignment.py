from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


@dataclass
class AlignmentResult:
    """Container for alignment outputs."""

    shifts: np.ndarray  # (Z, 2) or (Z, 3)
    aligned_volume: np.ndarray


class StackAligner:
    """Drift-corrected alignment of 2D stack into a 3D volume.

    Uses phase cross-correlation between each slice and a reference
    (by default, the first slice).
    """

    def __init__(self, upsample_factor: int = 10, reference_idx: int = 0):
        self.upsample_factor = upsample_factor
        self.reference_idx = reference_idx

    # -----------------------------
    # public API
    # -----------------------------
    def estimate_shifts(self, volume: np.ndarray) -> np.ndarray:
        """Estimate per-slice shifts.

        Parameters
        ----------
        volume:
            Input volume of shape (Z, Y, X).

        Returns
        -------
        np.ndarray
            Shifts of shape (Z, 2), where each row is (shift_y, shift_x)
            relative to the reference slice.
        """
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        z, _, _ = volume.shape
        shifts = np.zeros((z, 2), dtype=np.float32)

        ref = volume[self.reference_idx].astype(np.float32)

        for idx in range(z):
            if idx == self.reference_idx:
                continue

            moving = volume[idx].astype(np.float32)
            shift_vec, _, _ = phase_cross_correlation(
                ref,
                moving,
                upsample_factor=self.upsample_factor,
            )
            # shift_vec = (shift_y, shift_x)
            shifts[idx] = shift_vec[:2]

        return shifts

    def apply_shifts(self, volume: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """Apply shifts to correct drift.

        Parameters
        ----------
        volume:
            Input volume of shape (Z, Y, X).
        shifts:
            Array of shape (Z, 2) or (Z, 3).

        Returns
        -------
        np.ndarray
            Aligned volume.
        """
        if volume.shape[0] != shifts.shape[0]:
            raise ValueError("Number of shifts must match number of slices (Z).")

        aligned = np.empty_like(volume)
        for idx in range(volume.shape[0]):
            shift_vec = shifts[idx]
            # Only Y/X shift for 2D slices.
            dy, dx = float(shift_vec[0]), float(shift_vec[1])
            aligned[idx] = nd_shift(volume[idx], shift=(dy, dx), order=1, mode="nearest")
        return aligned

    def align(self, volume: np.ndarray) -> AlignmentResult:
        """Full pipeline: estimate + apply."""
        shifts = self.estimate_shifts(volume)
        aligned = self.apply_shifts(volume, shifts)
        return AlignmentResult(shifts=shifts, aligned_volume=aligned)