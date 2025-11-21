import numpy as np

from tomo_pipeline.segmentation import Segmenter


def test_segmentation_otsu_binary_mask():
    z, y, x = 8, 16, 16
    volume = np.zeros((z, y, x), dtype=np.float32)
    volume[:, 8:, :] = 1.0  # half bright, half dark

    seg = Segmenter(method="otsu")
    mask = seg.segment(volume)

    assert mask.shape == volume.shape
    assert mask.dtype == bool
    # Should segment roughly half the volume
    ratio = mask.mean()
    assert 0.3 < ratio < 0.7