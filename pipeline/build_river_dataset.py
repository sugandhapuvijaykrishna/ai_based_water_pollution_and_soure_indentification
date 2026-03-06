"""
River Dataset Builder
=====================

Builds a binary river/non-river dataset from the existing
`final_fusion_dataset.npz` produced by `fusion_dataset.py`.

Assumptions
-----------
- X_patches: (N, 32, 32, 6) with channels:
    [Green, Red, NIR, NIR_proxy, NDWI, NDTI]
- X_meta   : (N, 2) with [lat, lon]

Label definition
----------------
- Compute NDWI per patch (channel index 4).
- A patch is labelled "river" (1) if the fraction of pixels
  with NDWI > NDWI_THRESH is >= COVERAGE_THRESH.
- Otherwise it is labelled "non-river" (0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR.parent / "outputs"
SOURCE_NPZ = OUTPUTS_DIR / "final_fusion_dataset.npz"
RIVER_NPZ = OUTPUTS_DIR / "river_dataset.npz"

# NDWI > 0.1 is a common heuristic for water; tuneable
NDWI_THRESH = 0.10
# Require at least 60% of pixels to be water-like
COVERAGE_THRESH = 0.60


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("river_dataset")


def build_river_dataset() -> None:
    if not SOURCE_NPZ.exists():
        log.error("Source dataset not found: %s", SOURCE_NPZ)
        log.error("Run fusion_dataset.py first.")
        return

    log.info("Loading source dataset: %s", SOURCE_NPZ)
    # allow_pickle=True is required because `patch_info` was saved
    # as an object array in the original pipeline.
    data = np.load(SOURCE_NPZ, allow_pickle=True)

    X_patches = data["X_patches"].astype("float32")  # (N, 32, 32, 6)
    X_meta = data["X_meta"].astype("float32")        # (N, 2)
    patch_info = data.get("patch_info", None)

    if X_patches.ndim != 4 or X_patches.shape[-1] < 5:
        log.error("Unexpected X_patches shape: %s", X_patches.shape)
        return

    log.info("Patches shape: %s", X_patches.shape)

    # NDWI is channel index 4
    ndwi = X_patches[..., 4]
    # Fraction of pixels in each patch above threshold
    water_frac = (ndwi > NDWI_THRESH).mean(axis=(1, 2))

    y_river = (water_frac >= COVERAGE_THRESH).astype("int32")

    n = len(y_river)
    river_count = int(y_river.sum())
    nonriver_count = n - river_count

    log.info("Total patches: %d", n)
    log.info("River patches   (label=1): %d (%.1f%%)", river_count, 100.0 * river_count / n)
    log.info("Non-river patches (label=0): %d (%.1f%%)", nonriver_count, 100.0 * nonriver_count / n)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RIVER_NPZ,
        X_patches=X_patches,
        X_meta=X_meta,
        y_river=y_river,
        patch_info=patch_info if patch_info is not None else np.arange(n),
    )

    size_mb = RIVER_NPZ.stat().st_size / 1_048_576
    log.info("Saved river dataset to %s (%.1f MB)", RIVER_NPZ, size_mb)


if __name__ == "__main__":
    build_river_dataset()

