import numpy as np

from tomo_pipeline.alignment import StackAligner


def test_alignment_identity():
    # volume with no drift -> shifts should be ~0
    z, y, x = 5, 32, 32
    base = np.random.rand(y, x)
    volume = np.stack([base for _ in range(z)], axis=0)

    aligner = StackAligner(upsample_factor=1)
    shifts = aligner.estimate_shifts(volume)

    assert shifts.shape == (z, 2)
    assert np.allclose(shifts, 0, atol=0.1)