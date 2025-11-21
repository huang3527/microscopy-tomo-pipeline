from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from skimage import io as skio

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class StackLoader:
    """Load and save 3D volumes from 2D microscopy stacks.

    This class intentionally keeps the I/O layer simple:

    - If `path` is a directory, all image files are sorted by name and stacked.
    - If `path` is a single TIFF, it is read as a multi-page stack.
    - Saving can be `.npy` (NumPy array) or multi-page TIFF.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -----------------------------
    # public API
    # -----------------------------
    def load_stack(self) -> np.ndarray:
        """Return 3D volume as np.ndarray with shape (Z, Y, X)."""
        if self.path.is_dir():
            return self._load_from_folder(self.path)
        if self.path.suffix.lower() in (".tif", ".tiff"):
            return self._load_from_tiff(self.path)
        if self.path.suffix.lower() == ".npy":
            return np.load(self.path)
        raise ValueError(f"Unsupported input path: {self.path}")

    def save_stack(self, volume: np.ndarray, out_path: str | Path) -> None:
        """Save 3D volume for later processing.

        The format is inferred from the file extension:
        - `.npy`  -> NumPy binary
        - `.tif`/`.tiff` -> multi-page TIFF
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() == ".npy":
            np.save(out_path, volume)
            return

        if out_path.suffix.lower() in (".tif", ".tiff"):
            skio.imsave(out_path, volume.astype(np.float32))
            return

        raise ValueError(f"Unsupported output format: {out_path.suffix}")

    # -----------------------------
    # internal helpers
    # -----------------------------
    def _load_from_folder(self, folder: Path) -> np.ndarray:
        files: Sequence[Path] = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        )
        if not files:
            raise FileNotFoundError(f"No image files found in {folder}")

        stack = [skio.imread(f) for f in files]
        volume = np.stack(stack, axis=0)  # (Z, Y, X)
        return volume

    def _load_from_tiff(self, path: Path) -> np.ndarray:
        arr = skio.imread(path)
        # skimage.io.imread returns (Z, Y, X) for multi-page TIFF
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr