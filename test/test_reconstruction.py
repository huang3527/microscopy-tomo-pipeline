import numpy as np

from tomo_pipeline.reconstruction import VolumeReconstructor


def test_reconstruction_normalization_and_crop():
    z, y, x = 10, 20, 30
    volume = np.linspace(0, 100, z * y * x, dtype=np.float32).reshape(z, y, x)

    recon = VolumeReconstructor()
    out = recon.build_volume(volume, margin=2)

    assert out.ndim == 3
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    assert out.shape[0] == z - 4
    assert out.shape[1] == y - 4
    assert out.shape[2] == x - 4