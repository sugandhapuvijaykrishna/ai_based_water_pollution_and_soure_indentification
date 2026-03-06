"""
Synthetic In-Situ Sensor Data Generator – Krishna River
=========================================================
Generates ground_truth_sensors.csv (and per-month CSVs) for 5 virtual
sensor stations placed along the Krishna River.

Seasonal NTU ranges (Krishna River basin, India):
  Summer   (April, May)            : 400 – 800 NTU  (high concentration)
  Monsoon  (July, September)       : 100 – 300 NTU  (silt, noisy)
  Winter   (November, Dec)         : 10  – 50  NTU  (low, clear)

A distance-decay function reduces NTU as stations are farther from the
pollution source, teaching the model the direction of the plume.

Outputs
-------
  outputs/ground_truth_sensors.csv          ← all months combined
  outputs/<month_name>_sensors.csv          ← one per month folder
"""

import os
import math
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Pollution source coordinate (industrial outlet on Krishna River)
# Adjust this to the actual outfall point visible in your imagery.
# Default: near Vijayawada industrial corridor, Krishna delta upstream sector.
SOURCE_LAT = 16.5193
SOURCE_LON = 80.6305

# 5 Virtual sensor stations along the Krishna River (upstream → downstream)
# Stations are ordered roughly west→east following the river course.
STATIONS = [
    {"id": "KR-01", "name": "Amaravati_Upstream",   "lat": 16.5730, "lon": 80.3560},
    {"id": "KR-02", "name": "Industrial_Zone",       "lat": 16.5193, "lon": 80.6305},  # ← source
    {"id": "KR-03", "name": "Vijayawada_Mid",        "lat": 16.5100, "lon": 80.6200},
    {"id": "KR-04", "name": "Gannavaram_Downstream", "lat": 16.5430, "lon": 80.7980},
    {"id": "KR-05", "name": "Krishna_Delta_Entry",   "lat": 16.1200, "lon": 81.0800},
]

# Map output folder names → season
MONTH_SEASON = {
    "April":    "Summer",
    "May":      "Summer",
    "Jul":      "Monsoon",
    "Sept":     "Monsoon",
    "Nov":      "Winter",
    "Nov (2)":  "Winter",
    "Dec":      "Winter",
}

# Approximate calendar months for timestamp generation
MONTH_CALENDAR = {
    "April":    (2025, 4),
    "May":      (2025, 5),
    "Jul":      (2025, 7),
    "Sept":     (2025, 9),
    "Nov":      (2025, 11),
    "Nov (2)":  (2025, 11),
    "Dec":      (2025, 12),
}

# NTU ranges and noise σ per season
SEASON_CONFIG = {
    "Summer":  {"ntu_range": (400, 800), "noise_sigma": 15},
    "Monsoon": {"ntu_range": (100, 300), "noise_sigma": 45},   # high noise = silt
    "Winter":  {"ntu_range": (10,   50), "noise_sigma":  5},
}

# How many sensor readings per station per month (observations during the day)
READINGS_PER_STATION = 6       # every 4 hours → 00:00, 04:00, …, 20:00

# Decay model: NTU = base_NTU * exp(-k * distance_km)
# Larger k → faster decay with distance.
DECAY_K = 0.08

RANDOM_SEED = 42
OUTPUT_DIR  = Path(__file__).parent / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two geographic points (km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def decay_factor(dist_km: float) -> float:
    """Exponential decay: 1.0 at source, approaches 0 far away."""
    return math.exp(-DECAY_K * dist_km)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def generate_timestamps(year: int, month: int, n: int):
    """Return *n* equally spaced datetime strings within the given month."""
    base = datetime(year, month, 15, 0, 0, 0)          # mid-month
    step = timedelta(hours=24 // n)
    return [(base + step * i).strftime("%Y-%m-%dT%H:%M:%S") for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_monthly_records(month_label: str, rng: random.Random) -> list[dict]:
    """
    Generate sensor records for all stations for one month.

    Returns a list of dicts, one per (station × timestamp).
    """
    season  = MONTH_SEASON[month_label]
    cfg     = SEASON_CONFIG[season]
    ntu_lo, ntu_hi = cfg["ntu_range"]
    sigma   = cfg["noise_sigma"]
    year, month_num = MONTH_CALENDAR[month_label]

    records = []

    for station in STATIONS:
        dist_km = haversine_km(SOURCE_LAT, SOURCE_LON,
                               station["lat"], station["lon"])
        df      = decay_factor(dist_km)

        timestamps = generate_timestamps(year, month_num, READINGS_PER_STATION)

        for ts in timestamps:
            # Base NTU: random within seasonal range, scaled by decay
            base_ntu = rng.uniform(ntu_lo, ntu_hi) * df

            # Add Gaussian noise (monsoon is noisier → higher sigma)
            noise    = rng.gauss(0, sigma)
            ntu      = clamp(base_ntu + noise, 5, 1500)

            records.append({
                "Timestamp":     ts,
                "Month":         month_label,
                "Season":        season,
                "Station_ID":    station["id"],
                "Station_Name":  station["name"],
                "Lat":           station["lat"],
                "Lon":           station["lon"],
                "Dist_from_Source_km": round(dist_km, 3),
                "Decay_Factor":  round(df, 4),
                "Turbidity_NTU": round(ntu, 2),
            })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "Timestamp", "Month", "Season",
    "Station_ID", "Station_Name",
    "Lat", "Lon",
    "Dist_from_Source_km", "Decay_Factor",
    "Turbidity_NTU",
]


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  Saved → {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    all_records = []

    print(f"\nSynthetic Sensor Generator – Krishna River")
    print(f"Source coordinate : {SOURCE_LAT}°N  {SOURCE_LON}°E")
    print(f"Stations          : {len(STATIONS)}")
    print(f"Month folders     : {list(MONTH_SEASON.keys())}\n")

    # ── Per-month CSV (matches output folder names) ──────────────────────────
    for month_label in MONTH_SEASON:
        print(f"[{MONTH_SEASON[month_label]:8s}]  {month_label}")
        records = generate_monthly_records(month_label, rng)

        # Print station summary
        for s in STATIONS:
            s_rows = [r for r in records if r["Station_ID"] == s["id"]]
            ntus   = [r["Turbidity_NTU"] for r in s_rows]
            print(f"         {s['id']}  dist={s_rows[0]['Dist_from_Source_km']:6.1f} km  "
                  f"NTU avg={sum(ntus)/len(ntus):6.1f}  "
                  f"range=[{min(ntus):.1f}, {max(ntus):.1f}]")

        # Save individual month CSV  (e.g. April_sensors.csv)
        safe_name = month_label.replace(" ", "_")
        month_csv = os.path.join(OUTPUT_DIR, f"{safe_name}_sensors.csv")
        write_csv(month_csv, records)

        all_records.extend(records)
        print()

    # ── Combined ground-truth CSV ─────────────────────────────────────────────
    combined_csv = os.path.join(OUTPUT_DIR, "ground_truth_sensors.csv")
    write_csv(combined_csv, all_records)

    print(f"\n{'─'*55}")
    print(f"Total records written : {len(all_records)}")
    print(f"Combined CSV          : {combined_csv}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
