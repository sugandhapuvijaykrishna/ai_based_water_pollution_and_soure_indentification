"""
Synthetic Water Pollution Data Generator  –  Sentinel-2 per-band edition
=========================================================================
Handles the real Copernicus/SentinelHub folder layout where each band
arrives as a separate TIFF file:

    <Month>/
        *_B03_(Raw).tiff   ←  Green
        *_B04_(Raw).tiff   ←  Red
        *_B08_(Raw).tiff   ←  NIR

Pipeline per month
──────────────────
1. Discover & stack B03 / B04 / B08 into a 3-band in-memory array
2. Compute NDWI water mask   (Green − NIR) / (Green + NIR) > 0.3
3. Build anisotropic Gaussian plume from an outlet point
4. Inject turbidity spectral signal  (Red ×1.20-1.40, NIR ×1.30-1.50)
5. Save  <month>_polluted.tif   (3 bands, float32)
6. Save  <month>_labels.tif     (uint8: 0=clean, 1=moderate, 2=high)

Requirements:
    pip install rasterio numpy scipy
"""

import os
import glob
import logging
import numpy as np
import rasterio
from rasterio import MemoryFile
from rasterio.transform import rowcol
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
NDWI_THRESHOLD           = 0.3    # NDWI > threshold  →  water pixel
PLUME_SIGMA_X            = 80     # cross-stream Gaussian spread (pixels)
PLUME_SIGMA_Y            = 50     # along-stream Gaussian spread (pixels)
HIGH_POLLUTION_THRESH    = 0.55   # normalised plume value → class 2
MODERATE_POLLUTION_THRESH= 0.20   # normalised plume value → class 1
RED_FACTOR_RANGE         = (1.20, 1.40)
NIR_FACTOR_RANGE         = (1.30, 1.50)

# Band filename fragments (case-insensitive glob)
BAND_PATTERNS = {
    "green": "*_B03_*.tiff",
    "red":   "*_B04_*.tiff",
    "nir":   "*_B08_*.tiff",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find a single band file in a folder
# ─────────────────────────────────────────────────────────────────────────────
def _find_band(folder: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in '{folder}'"
        )
    if len(matches) > 1:
        log.warning("Multiple matches for '%s' – using first: %s",
                    pattern, matches[0])
    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Stack Bands
# ─────────────────────────────────────────────────────────────────────────────
def load_bands(folder: str) -> tuple:
    """
    Load B03 (Green), B04 (Red), B08 (NIR) from *folder*.
    Returns (green, red, nir) as float32 arrays plus the rasterio profile
    and transform from the Green band (reference).
    """
    path_g = _find_band(folder, BAND_PATTERNS["green"])
    path_r = _find_band(folder, BAND_PATTERNS["red"])
    path_n = _find_band(folder, BAND_PATTERNS["nir"])

    log.info("  Green : %s", os.path.basename(path_g))
    log.info("  Red   : %s", os.path.basename(path_r))
    log.info("  NIR   : %s", os.path.basename(path_n))

    def _read(p):
        with rasterio.open(p) as src:
            return src.read(1).astype(np.float32), src.profile, src.transform

    green, profile, transform = _read(path_g)
    red,   _,       _         = _read(path_r)
    nir,   _,       _         = _read(path_n)

    return green, red, nir, profile, transform


# ─────────────────────────────────────────────────────────────────────────────
# 2. NDWI Water Mask
# ─────────────────────────────────────────────────────────────────────────────
def compute_ndwi_mask(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    denom = np.where(denom == 0, np.finfo(np.float32).eps, denom)
    ndwi  = (green - nir) / denom
    return ndwi > NDWI_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gaussian Plume
# ─────────────────────────────────────────────────────────────────────────────
def make_gaussian_plume(shape: tuple, source_rc: tuple,
                        water_mask: np.ndarray) -> np.ndarray:
    rows, cols = shape
    src_r, src_c = source_rc
    r_grid, c_grid = np.mgrid[0:rows, 0:cols]

    plume = np.exp(
        -(((c_grid - src_c) ** 2) / (2 * PLUME_SIGMA_X ** 2) +
          ((r_grid - src_r) ** 2) / (2 * PLUME_SIGMA_Y ** 2))
    )
    plume = gaussian_filter(plume, sigma=4)          # natural diffusion blur
    p_max = plume.max()
    if p_max > 0:
        plume /= p_max
    plume = np.where(water_mask, plume, 0.0)         # confine to water
    return plume.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spectral Injection
# ─────────────────────────────────────────────────────────────────────────────
def inject_pollution(green, red, nir, plume, rng):
    red_factor = rng.uniform(*RED_FACTOR_RANGE)
    nir_factor = rng.uniform(*NIR_FACTOR_RANGE)
    red_mod = np.clip(red * (1.0 + (red_factor - 1.0) * plume), 0, 65535)
    nir_mod = np.clip(nir * (1.0 + (nir_factor - 1.0) * plume), 0, 65535)
    return green.copy(), red_mod, nir_mod


# ─────────────────────────────────────────────────────────────────────────────
# 5. Label Mask
# ─────────────────────────────────────────────────────────────────────────────
def make_labels(plume: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    label = np.zeros(plume.shape, dtype=np.uint8)
    label[water_mask & (plume >= MODERATE_POLLUTION_THRESH)] = 1
    label[water_mask & (plume >= HIGH_POLLUTION_THRESH)]     = 2
    return label


# ─────────────────────────────────────────────────────────────────────────────
# 6. Auto-detect outlet: nearest water pixel to image centre
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_outlet(water_mask: np.ndarray,
                        rng: np.random.Generator) -> tuple:
    rows, cols = water_mask.shape
    water_coords = np.argwhere(water_mask)
    if len(water_coords) == 0:
        return rows // 2, cols // 2
    centre = np.array([rows // 2, cols // 2])
    dists  = np.linalg.norm(water_coords - centre, axis=1)
    # Pick randomly among the closest 5 % to add variation per month
    top_n  = max(1, len(dists) // 20)
    closest_idx = np.argpartition(dists, min(top_n, len(dists)-1))[:top_n]
    chosen = water_coords[rng.choice(closest_idx)]
    return int(chosen[0]), int(chosen[1])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Process one month folder
# ─────────────────────────────────────────────────────────────────────────────
def process_month_folder(folder: str,
                          output_dir: str,
                          outlet_lonlat: tuple = None,
                          seed: int = 42) -> None:
    month_name = os.path.basename(folder.rstrip("/\\"))
    log.info("=" * 60)
    log.info("Month: %s", month_name)
    rng = np.random.default_rng(seed)

    # ── Load ──────────────────────────────────────────────────
    try:
        green, red, nir, profile, transform = load_bands(folder)
    except FileNotFoundError as e:
        log.error("  %s – skipping.", e)
        return

    rows, cols = green.shape

    # ── Water mask ─────────────────────────────────────────────
    water_mask = compute_ndwi_mask(green, nir)
    water_pct  = water_mask.mean() * 100
    log.info("  Water coverage : %.1f %%", water_pct)

    # ── Outlet pixel ───────────────────────────────────────────
    if outlet_lonlat is not None:
        try:
            src_r, src_c = rowcol(transform, outlet_lonlat[0], outlet_lonlat[1])
            src_r = int(np.clip(src_r, 0, rows - 1))
            src_c = int(np.clip(src_c, 0, cols - 1))
            log.info("  Outlet (geo)   : lon=%.4f lat=%.4f → (%d, %d)",
                     *outlet_lonlat, src_r, src_c)
        except Exception as e:
            log.warning("  Outlet conversion failed: %s → auto-detect.", e)
            src_r, src_c = auto_detect_outlet(water_mask, rng)
    else:
        src_r, src_c = auto_detect_outlet(water_mask, rng)
        log.info("  Outlet (auto)  : pixel (%d, %d)", src_r, src_c)

    # ── Plume ──────────────────────────────────────────────────
    plume = make_gaussian_plume((rows, cols), (src_r, src_c), water_mask)

    # ── Spectral injection ─────────────────────────────────────
    green_m, red_m, nir_m = inject_pollution(green, red, nir, plume, rng)

    # ── Labels ─────────────────────────────────────────────────
    labels = make_labels(plume, water_mask)
    log.info("  Labels → high=%d px  moderate=%d px  clean=%d px",
             (labels == 2).sum(), (labels == 1).sum(), (labels == 0).sum())

    os.makedirs(output_dir, exist_ok=True)

    # ── Save polluted image ────────────────────────────────────
    polluted_path = os.path.join(output_dir, f"{month_name}_polluted.tif")
    p_meta = profile.copy()
    p_meta.update(count=3, dtype="float32", compress="lzw")
    with rasterio.open(polluted_path, "w", **p_meta) as dst:
        dst.write(green_m, 1)
        dst.write(red_m,   2)
        dst.write(nir_m,   3)
        dst.update_tags(
            BAND_1="Green (B03)", BAND_2="Red (B04)", BAND_3="NIR (B08)",
            NDWI_THRESHOLD=str(NDWI_THRESHOLD),
        )
    log.info("  Saved → %s", polluted_path)

    # ── Save label mask ────────────────────────────────────────
    label_path = os.path.join(output_dir, f"{month_name}_labels.tif")
    l_meta = profile.copy()
    l_meta.update(count=1, dtype="uint8", compress="lzw", nodata=255)
    with rasterio.open(label_path, "w", **l_meta) as dst:
        dst.write(labels, 1)
        dst.update_tags(CLASSES="0=Clean, 1=Moderate, 2=High")
    log.info("  Saved → %s", label_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Batch loop over all month sub-folders
# ─────────────────────────────────────────────────────────────────────────────
def process_all_months(root_dir: str,
                        output_dir: str,
                        outlet_lonlat: tuple = None,
                        seed: int = 42) -> None:
    """
    Walk every immediate sub-folder of *root_dir* (each representing a month)
    and run the full synthetic pollution pipeline on it.
    """
    subdirs = sorted([
        d for d in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(d)
    ])
    if not subdirs:
        log.error("No sub-folders found in: %s", root_dir)
        return

    log.info("Found %d month folder(s): %s",
             len(subdirs), [os.path.basename(d) for d in subdirs])

    for idx, folder in enumerate(subdirs):
        process_month_folder(
            folder=folder,
            output_dir=output_dir,
            outlet_lonlat=outlet_lonlat,
            seed=seed + idx,
        )

    log.info("=" * 60)
    log.info("All months processed. Outputs saved to: %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic water pollution GeoTIFFs from "
                    "Sentinel-2 per-band monthly folders."
    )
    parser.add_argument("root_dir",   help="Parent folder containing month sub-folders")
    parser.add_argument("output_dir", help="Destination folder for output TIFFs")
    parser.add_argument("--lon", type=float, default=None,
                        help="Longitude of industrial outlet (WGS84)")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude of industrial outlet (WGS84)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    args = parser.parse_args()

    outlet = (args.lon, args.lat) if (args.lon is not None and
                                       args.lat is not None) else None
    process_all_months(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        outlet_lonlat=outlet,
        seed=args.seed,
    )
