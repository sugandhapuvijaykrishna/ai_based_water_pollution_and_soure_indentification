"""
Fusion Dataset Builder
======================
Builds 'final_fusion_dataset.npz' by linking 32×32 image patches extracted
from monthly polluted Sentinel-2 GeoTIFFs to the nearest virtual sensor
reading (Turbidity NTU) using GPS coordinates.

Folder layout expected
──────────────────────
  <root>/outputs/
      April_polluted.tif        ← 3-band spectral image
      April_labels.tif          ← uint8 label raster  (optional, saved in npz)
      April_sensors.csv
      May_polluted.tif
      ...
      ground_truth_sensors.csv  ← (not used directly; per-month CSVs are used)

NPZ contents
────────────
  X_patches  : float32  (N, 32, 32, 6)   image patch  [B3,B4,B8,NDWI,NDTI,NIR_norm]
  X_meta     : float32  (N, 2)            [centre_lat, centre_lon]
  y_ntu      : float32  (N,)              Turbidity NTU from nearest sensor
  y_label    : int32    (N,)              label class 0/1/2 (majority vote in patch)
  patch_info : (N,) object array          strings "month|row|col|station_id"

Adapter note
────────────
preprocessing.LandsatProcessor expects B3/B4/B5/B6 band names.
Sentinel-2 has B03(Green)/B04(Red)/B08(NIR) — no SWIR.
We map:  B3→Green  B4→Red  B5→NIR  B6≡NIR (proxy for absent SWIR1)
NDTI then becomes (Red−NIR)/(Red+NIR) which approximates a turbidity index
for water bodies — scientifically acceptable for inland water quality work.
"""

import os
import sys
import csv
import math
import logging
import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from pathlib import Path
from scipy.ndimage import binary_erosion, binary_dilation

# ── Resolve Model/ directory on the path ─────────────────────────────────────
ROOT_DIR  = Path(__file__).parent
MODEL_DIR = ROOT_DIR.parent / "model_core"
sys.path.insert(0, str(MODEL_DIR))

from preprocessing import LandsatProcessor        # noqa: E402
from patch_extraction import PatchExtractor        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR   = ROOT_DIR.parent / "outputs"
NPZ_OUT       = OUTPUTS_DIR / "final_fusion_dataset.npz"
PATCH_SIZE    = 32
OVERLAP_RATIO = 0.5          # 50 % overlap → stride = 16 px
LABEL_METHOD  = "majority"   # 'majority' or 'center'

# Map output-folder prefix → CSV name (without _sensors.csv suffix)
MONTH_MAP = {
    "April":    "April",
    "May":      "May",
    "Jul":      "Jul",
    "Sept":     "Sept",
    "Nov":      "Nov",
    "Nov (2)":  "Nov__2_",    # spaces→_ as written by generate_sensor_data.py
    "Dec":      "Dec",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-2 Adapter for LandsatProcessor
# ─────────────────────────────────────────────────────────────────────────────
class Sentinel2Adapter(LandsatProcessor):
    """
    Reads a 3-band Sentinel-2 GeoTIFF (Green/Red/NIR) and uses
    LandsatProcessor.create_feature_stack() to build the 6-channel stack.

    Band mapping (Sentinel-2 → Landsat slot):
        B3 (Green)  → Green slot
        B4 (Red)    → Red   slot
        B8 (NIR)    → NIR   slot
        B8 (NIR)    → SWIR1 slot  [proxy – no SWIR in our dataset]

    Resulting channels: [Green, Red, NIR, NIR_proxy, NDWI, NDTI]
    NDWI = (Green−NIR)/(Green+NIR)   — water index
    NDTI = (Red−NIR)/(Red+NIR)       — turbidity proxy
    """

    def load_sentinel2_tiff(self, tiff_path: str):
        """
        Load a 3-band Sentinel-2 polluted GeoTIFF and return:
            feature_stack  (H, W, 6)   float32
            transform      rasterio Affine transform
            crs            rasterio CRS
        """
        path = Path(tiff_path)
        if not path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {path}")

        with rasterio.open(path) as src:
            green = src.read(1).astype(np.float32)
            red   = src.read(2).astype(np.float32)
            nir   = src.read(3).astype(np.float32)
            transform = src.transform
            crs       = src.crs
            self.profile = src.profile

        # Write four individual band TIFFs into a temp memory structure
        # so we can reuse LandsatProcessor.create_feature_stack() unchanged.
        # We write to a shared in-memory dict instead of disk.
        self._s2_bands = {
            "B3": green,
            "B4": red,
            "B5": nir,
            "B6": nir,   # NIR proxy for SWIR1
        }

        # Build feature stack directly (bypass file-based load_band)
        features, meta = self._create_stack_from_arrays(
            green, red, nir, nir, normalize=True
        )
        return features, transform, crs, meta

    def _create_stack_from_arrays(self, green, red, nir, swir_proxy,
                                  normalize: bool = True):
        """Mirror of create_feature_stack() but from in-memory arrays."""
        if normalize:
            green     = self.normalize_band(green)
            red       = self.normalize_band(red)
            nir       = self.normalize_band(nir)
            swir_proxy = self.normalize_band(swir_proxy)

        ndwi = self.calculate_ndwi(green, nir)
        ndti = self.calculate_ndti(red, swir_proxy)

        features = np.stack([green, red, nir, swir_proxy, ndwi, ndti], axis=-1)

        meta = {
            "shape":    features.shape,
            "channels": ["B3_Green", "B4_Red", "B8_NIR", "B8_NIR_proxy",
                         "NDWI", "NDTI"],
        }
        return features.astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────────────
# Sensor CSV Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_sensor_csv(csv_path: str) -> list[dict]:
    """Return list of dicts from a month sensor CSV."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            rows.append({
                "station_id": rec["Station_ID"],
                "lat":  float(rec["Lat"]),
                "lon":  float(rec["Lon"]),
                "ntu":  float(rec["Turbidity_NTU"]),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Spatial linking helpers
# ─────────────────────────────────────────────────────────────────────────────
def pixel_to_latlon(transform, row: int, col: int) -> tuple[float, float]:
    """Convert pixel (row, col) → (lat, lon) using rasterio transform."""
    # transform_xy returns (x, y) → (lon, lat) for geographic CRS
    x, y = transform_xy(transform, row, col, offset="center")
    return float(y), float(x)   # (lat, lon)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_sensor(lat: float, lon: float,
                   sensors: list[dict]) -> dict:
    """Return the sensor record closest to (lat, lon)."""
    best, best_d = None, float("inf")
    for s in sensors:
        d = haversine_km(lat, lon, s["lat"], s["lon"])
        if d < best_d:
            best_d, best = d, s
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Per-month processor
# ─────────────────────────────────────────────────────────────────────────────
def process_month(month_label: str,
                  polluted_tif: Path,
                  label_tif: Path,
                  sensor_csv: Path,
                  extractor: PatchExtractor) -> dict:
    """
    Returns dict with keys: patches, meta_latlon, ntu, labels, info
    """
    log.info("=" * 60)
    log.info("Month: %s", month_label)

    # ── 1. Load & build 6-channel feature stack ────────────────────────────
    adapter = Sentinel2Adapter(str(polluted_tif.parent))
    features, geo_transform, crs, stack_meta = \
        adapter.load_sentinel2_tiff(str(polluted_tif))

    log.info("  Feature stack : %s  channels=%s",
             features.shape, stack_meta["channels"])

    # ── 2. Extract 32×32 patches from feature stack ────────────────────────
    # NDWI water mask: channel index 4 is NDWI in the 6-channel stack
    ndwi_channel = features[:, :, 4]
    # Lower NDWI threshold to include more water-edge patches for better label variety
    water_mask = ndwi_channel > 0.10  

    # Remove isolated pixels (noise)
    water_mask = binary_erosion(water_mask, iterations=2)
    # Restore river body size after erosion
    water_mask = binary_dilation(water_mask, iterations=2)

    patches, coords = extractor.extract_patches(
        features, skip_nodata=True, mask=water_mask
    )
    log.info("  Patches extracted: %d", len(patches))

    if len(patches) == 0:
        log.warning("  No valid patches – skipping month.")
        return {}

    # ── 3. Load label raster & assign per-patch label ──────────────────────
    if label_tif.exists():
        labels_arr = extractor.load_labels_and_extract(
            str(label_tif), coords, method=LABEL_METHOD
        )
    else:
        log.warning("  Label TIF not found – assigning label=0 to all patches.")
        labels_arr = np.zeros(len(patches), dtype=np.int32)

    # Align: load_labels_and_extract may skip invalid patches (returns fewer)
    # Trimming patches/coords to match
    n = min(len(patches), len(labels_arr))
    patches    = patches[:n]
    coords     = coords[:n]
    labels_arr = labels_arr[:n]

    # ── 4. Load sensor data ────────────────────────────────────────────────
    sensors = load_sensor_csv(str(sensor_csv))
    log.info("  Sensor readings loaded: %d", len(sensors))

    # ── 5. Spatial link: centre pixel → (lat,lon) → nearest sensor NTU ────
    centre_half = PATCH_SIZE // 2
    latlon_list   = []
    ntu_list      = []
    info_list     = []

    for (row, col) in coords:
        ctr_row = row + centre_half
        ctr_col = col + centre_half
        lat, lon = pixel_to_latlon(geo_transform, ctr_row, ctr_col)
        sensor   = nearest_sensor(lat, lon, sensors)
        latlon_list.append([lat, lon])
        ntu_list.append(sensor["ntu"])
        info_list.append(f"{month_label}|{row}|{col}|{sensor['station_id']}")

    meta_arr  = np.array(latlon_list,  dtype=np.float32)   # (N, 2)
    ntu_arr   = np.array(ntu_list,     dtype=np.float32)   # (N,)
    info_arr  = np.array(info_list,    dtype=object)       # (N,)

    log.info("  NTU stats → min=%.1f  max=%.1f  mean=%.1f",
             ntu_arr.min(), ntu_arr.max(), ntu_arr.mean())

    return {
        "patches": patches,           # (N, 32, 32, 6)
        "meta":    meta_arr,          # (N, 2)  [lat, lon]
        "ntu":     ntu_arr,           # (N,)
        "labels":  labels_arr,        # (N,)
        "info":    info_arr,          # (N,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build & Save
# ─────────────────────────────────────────────────────────────────────────────
def build_fusion_dataset():
    extractor = PatchExtractor(patch_size=PATCH_SIZE,
                               overlap_ratio=OVERLAP_RATIO)

    all_patches = []
    all_meta    = []
    all_ntu     = []
    all_labels  = []
    all_info    = []

    # ── Discover months automatically from outputs/ ────────────────────────
    month_labels_found = []
    for month_label, csv_prefix in MONTH_MAP.items():
        polluted_tif = OUTPUTS_DIR / f"{month_label}_polluted.tif"
        label_tif    = OUTPUTS_DIR / f"{month_label}_labels.tif"
        sensor_csv   = OUTPUTS_DIR / f"{csv_prefix}_sensors.csv"

        if not polluted_tif.exists():
            log.warning("Missing polluted TIF for '%s' – skipping.", month_label)
            continue
        if not sensor_csv.exists():
            log.warning("Missing sensor CSV for '%s' – skipping.", month_label)
            continue

        month_labels_found.append(month_label)
        result = process_month(
            month_label=month_label,
            polluted_tif=polluted_tif,
            label_tif=label_tif,
            sensor_csv=sensor_csv,
            extractor=extractor,
        )

        if not result:
            continue

        all_patches.append(result["patches"])
        all_meta.append(result["meta"])
        all_ntu.append(result["ntu"])
        all_labels.append(result["labels"])
        all_info.append(result["info"])

    if not all_patches:
        log.error("No data collected. Check your outputs/ folder paths.")
        return

    # ── Concatenate ────────────────────────────────────────────────────────
    X_patches = np.concatenate(all_patches, axis=0)   # (N, 32, 32, 6)
    X_meta    = np.concatenate(all_meta,    axis=0)   # (N, 2)
    y_ntu     = np.concatenate(all_ntu,     axis=0)   # (N,)
    y_label   = np.concatenate(all_labels,  axis=0)   # (N,)
    info      = np.concatenate(all_info,    axis=0)   # (N,)

    log.info("=" * 60)
    log.info("FINAL DATASET")
    log.info("  X_patches : %s  dtype=%s", X_patches.shape, X_patches.dtype)
    log.info("  X_meta    : %s  dtype=%s", X_meta.shape,    X_meta.dtype)
    log.info("  y_ntu     : %s  dtype=%s", y_ntu.shape,     y_ntu.dtype)
    log.info("  y_label   : %s  dtype=%s", y_label.shape,   y_label.dtype)
    log.info("  NTU  →  min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
             y_ntu.min(), y_ntu.max(), y_ntu.mean(), y_ntu.std())
    unique, counts = np.unique(y_label, return_counts=True)
    for u, c in zip(unique, counts):
        names = {0: "Clean", 1: "Moderate", 2: "High"}
        log.info("  Label %d (%s): %d patches (%.1f %%)",
                 u, names.get(u, "?"), c, 100 * c / len(y_label))

    # ── Save .npz ──────────────────────────────────────────────────────────
    np.savez_compressed(
        str(NPZ_OUT),
        X_patches  = X_patches,
        X_meta     = X_meta,
        y_ntu      = y_ntu,
        y_label    = y_label,
        patch_info = info,
    )
    size_mb = NPZ_OUT.stat().st_size / 1_048_576
    log.info("=" * 60)
    log.info("Saved → %s  (%.1f MB)", NPZ_OUT, size_mb)
    log.info("Months processed : %s", month_labels_found)
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_fusion_dataset()
