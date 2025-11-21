# microscopy-tomo-pipeline

A small, generic toolbox for building microscopy tomography pipelines in Python.

The goal is to provide **simple, reproducible building blocks** for:

- Loading 2D image stacks (FIB/SEM/TEM, or any other modality)
- Denoising and basic pre-processing
- Drift-corrected stack alignment
- Simple segmentation (e.g. Otsu threshold)
- Basic volume reconstruction helpers
- Lightweight visualization (orthogonal slices)

Everything is written in a **modalitiy-agnostic** way – there is no reference to
any specific company, tool vendor, or internal infrastructure.

---

## Installation

```bash
pip install -e .