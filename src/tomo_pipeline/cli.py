from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import StackAligner
from .filters import Denoiser
from .io import StackLoader
from .reconstruction import VolumeReconstructor


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tomography pipeline CLI")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder, multi-page TIFF, or .npy stack",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy or .tiff volume path",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for denoising (default: 1.0)",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Margin to crop from each side after alignment.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1) Load
    loader = StackLoader(input_path)
    volume = loader.load_stack()

    # 2) Denoise
    denoiser = Denoiser()
    volume = denoiser.gaussian(volume, sigma=args.sigma)

    # 3) Align
    aligner = StackAligner()
    result = aligner.align(volume)

    # 4) Reconstruct
    recon = VolumeReconstructor()
    final = recon.build_volume(result.aligned_volume, margin=args.margin)

    # 5) Save
    loader.save_stack(final, output_path)


if __name__ == "__main__":
    main()