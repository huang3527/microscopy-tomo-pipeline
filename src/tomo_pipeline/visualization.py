from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


class VolumeViewer:
    """Slicing + simple interactive views using matplotlib.

    This is deliberately lightweight – the goal is to give quick
    sanity-check views. More advanced 3D visualization (VTK/pyvista)
    can be added later as an optional backend.
    """

    def show_slice(self, volume: np.ndarray, axis: str = "z", index: int = 0):
        """Show a single slice along the chosen axis.

        Parameters
        ----------
        volume:
            Input volume (Z, Y, X).
        axis:
            One of {"z", "y", "x"}.
        index:
            Slice index along the chosen axis.
        """
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        axis = axis.lower()
        if axis not in {"z", "y", "x"}:
            raise ValueError("axis must be one of {'z', 'y', 'x'}.")

        if axis == "z":
            idx = np.clip(index, 0, volume.shape[0] - 1)
            img = volume[idx]
        elif axis == "y":
            idx = np.clip(index, 0, volume.shape[1] - 1)
            img = volume[:, idx, :]
        else:  # "x"
            idx = np.clip(index, 0, volume.shape[2] - 1)
            img = volume[:, :, idx]

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"Slice axis={axis}, index={idx}")
        plt.show()

    def orthoslices(self, volume: np.ndarray, x: int, y: int, z: int):
        """Show orthogonal slices through a given (z, y, x) location."""
        if volume.ndim != 3:
            raise ValueError("Expected a 3D volume (Z, Y, X).")

        z = int(np.clip(z, 0, volume.shape[0] - 1))
        y = int(np.clip(y, 0, volume.shape[1] - 1))
        x = int(np.clip(x, 0, volume.shape[2] - 1))

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(volume[z], cmap="gray")
        axes[0].set_title(f"Z={z}")
        axes[0].axis("off")

        axes[1].imshow(volume[:, y, :], cmap="gray")
        axes[1].set_title(f"Y={y}")
        axes[1].axis("off")

        axes[2].imshow(volume[:, :, x], cmap="gray")
        axes[2].set_title(f"X={x}")
        axes[2].axis("off")

        fig.tight_layout()
        plt.show()