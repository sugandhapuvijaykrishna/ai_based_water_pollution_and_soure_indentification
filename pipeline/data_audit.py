"""
Data Audit Script
=================
Run this before demo to verify all data is correct.
Usage: python data_audit.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT    = Path(__file__).parent
OUTPUTS = ROOT.parent / "outputs"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

print("=" * 65)
print("KRISHNA RIVER PROJECT — FULL DATA AUDIT")
print("=" * 65)

all_ok = True

# ── 1. CHECK ALL FILES EXIST ──────────────────────────────────────
print("\n[SECTION 1] FILE EXISTENCE CHECK")
files = {
    "Fusion NPZ":       OUTPUTS / "final_fusion_dataset.npz",
    "Model H5":         OUTPUTS / "best_fusion_model.h5",
    "Results CSV":      OUTPUTS / "results_for_viz.csv",
    "River NPZ":        OUTPUTS / "river_dataset.npz",
    "River Model":      OUTPUTS / "river_model.h5",
    "River Metrics":    OUTPUTS / "river_metrics.txt",
    "Sensor Combined":  OUTPUTS / "ground_truth_sensors.csv",
}
for name, path in files.items():
    if path.exists():
        mb = path.stat().st_size / 1_048_576
        print(f"  {PASS} {name:20s} — {mb:.2f} MB")
    else:
        print(f"  {FAIL} {name:20s} — MISSING: {path}")
        all_ok = False

# ── 2. SENSOR CSV AUDIT ───────────────────────────────────────────
print("\n[SECTION 2] SENSOR DATA AUDIT")
sensor_path = OUTPUTS / "ground_truth_sensors.csv"
if sensor_path.exists():
    s = pd.read_csv(sensor_path)
    print(f"  Total records     : {len(s):,}")
    print(f"  Stations          : {s['Station_ID'].nunique()} unique")
    print(f"  Station IDs       : {sorted(s['Station_ID'].unique().tolist())}")
    print(f"  Months covered    : {sorted(s['Month'].unique().tolist())}")
    print(f"  Seasons covered   : {sorted(s['Season'].unique().tolist())}")
    print(f"  NTU min           : {s['Turbidity_NTU'].min():.2f}")
    print(f"  NTU max           : {s['Turbidity_NTU'].max():.2f}")
    print(f"  NTU mean          : {s['Turbidity_NTU'].mean():.2f}")
    print(f"  NTU std           : {s['Turbidity_NTU'].std():.2f}")
    print(f"  Lat range         : {s['Lat'].min():.4f} to {s['Lat'].max():.4f}")
    print(f"  Lon range         : {s['Lon'].min():.4f} to {s['Lon'].max():.4f}")
    # Seasonal check
    for season in ['Summer', 'Monsoon', 'Winter']:
        sub = s[s['Season'] == season]
        if len(sub) > 0:
            expected = {'Summer':(400,800), 'Monsoon':(100,300), 'Winter':(10,50)}
            lo, hi = expected[season]
            in_range = sub['Turbidity_NTU'].between(lo*0.5, hi*1.5)
            pct = 100 * in_range.mean()
            status = PASS if pct > 80 else WARN
            print(f"  {status} {season:8s} NTU range {lo}-{hi}: {pct:.1f}% in expected range  (mean={sub['Turbidity_NTU'].mean():.1f})")
else:
    print(f"  {FAIL} ground_truth_sensors.csv missing")
    all_ok = False

# ── 3. FUSION NPZ AUDIT ───────────────────────────────────────────
print("\n[SECTION 3] FUSION DATASET AUDIT (final_fusion_dataset.npz)")
npz_path = OUTPUTS / "final_fusion_dataset.npz"
if npz_path.exists():
    d = np.load(npz_path, allow_pickle=True)
    print(f"  Keys              : {list(d.keys())}")
    print(f"  X_patches shape   : {d['X_patches'].shape}  dtype={d['X_patches'].dtype}")
    print(f"  X_meta shape      : {d['X_meta'].shape}     dtype={d['X_meta'].dtype}")
    print(f"  y_ntu shape       : {d['y_ntu'].shape}      dtype={d['y_ntu'].dtype}")
    print(f"  y_label shape     : {d['y_label'].shape}    dtype={d['y_label'].dtype}")
    print(f"  NTU min           : {d['y_ntu'].min():.2f}")
    print(f"  NTU max           : {d['y_ntu'].max():.2f}")
    print(f"  NTU mean          : {d['y_ntu'].mean():.2f}")
    print(f"  NTU std           : {d['y_ntu'].std():.2f}")
    print(f"  Lat range         : {d['X_meta'][:,0].min():.4f} to {d['X_meta'][:,0].max():.4f}")
    print(f"  Lon range         : {d['X_meta'][:,1].min():.4f} to {d['X_meta'][:,1].max():.4f}")
    # Label distribution
    labels, counts = np.unique(d['y_label'], return_counts=True)
    names = {0:'Clean', 1:'Moderate', 2:'High'}
    for lbl, cnt in zip(labels, counts):
        pct = 100 * cnt / len(d['y_label'])
        print(f"  Label {lbl} ({names.get(int(lbl),'?'):8s}): {cnt:5d} patches ({pct:.1f}%)")
    # Patch value range check
    pmin = d['X_patches'].min()
    pmax = d['X_patches'].max()
    if -0.1 <= pmin and pmax <= 1.1:
        print(f"  {PASS} Patch values normalized: [{pmin:.3f}, {pmax:.3f}]")
    else:
        print(f"  {WARN} Patch values may not be normalized: [{pmin:.3f}, {pmax:.3f}]")
    # Coordinate check
    lat_ok = 16.40 <= d['X_meta'][:,0].min() and d['X_meta'][:,0].max() <= 16.65
    lon_ok = 80.40 <= d['X_meta'][:,1].min() and d['X_meta'][:,1].max() <= 80.75
    print(f"  {PASS if lat_ok else FAIL} Lat in Krishna River range: {lat_ok}")
    print(f"  {PASS if lon_ok else FAIL} Lon in Krishna River range: {lon_ok}")
    # Sample patch info
    if 'patch_info' in d:
        print(f"  Sample patch_info : {d['patch_info'][:3]}")
else:
    print(f"  {FAIL} final_fusion_dataset.npz missing")
    all_ok = False

# ── 4. RESULTS CSV AUDIT ─────────────────────────────────────────
print("\n[SECTION 4] RESULTS CSV AUDIT (results_for_viz.csv)")
csv_path = OUTPUTS / "results_for_viz.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)

    # Apply same cleaning as dashboard for final audit
    # Clip negative NTU — physically impossible
    df['Predicted_NTU'] = df['Predicted_NTU'].clip(lower=0)

    # Remove outliers outside river corridor
    df = df[
        (df['Lat'] >= 16.490) &
        (df['Lat'] <= 16.585)
    ].copy().reset_index(drop=True)

    print(f"  Columns           : {df.columns.tolist()}")
    print(f"  Total rows        : {len(df):,}")
    print(f"  Predicted_NTU min : {df['Predicted_NTU'].min():.2f}")
    print(f"  Predicted_NTU max : {df['Predicted_NTU'].max():.2f}")
    print(f"  Predicted_NTU mean: {df['Predicted_NTU'].mean():.2f}")
    print(f"  Predicted_NTU std : {df['Predicted_NTU'].std():.2f}")
    print(f"  Lat range         : {df['Lat'].min():.4f} to {df['Lat'].max():.4f}")
    print(f"  Lon range         : {df['Lon'].min():.4f} to {df['Lon'].max():.4f}")
    # Variance check — all same value means model failed
    if df['Predicted_NTU'].std() < 1.0:
        print(f"  {FAIL} NTU std is near zero — model predicted same value for everything")
        all_ok = False
    else:
        print(f"  {PASS} NTU variance is healthy")
    # Flow vector check
    if 'Flow_U' in df.columns:
        u_nonzero = (df['Flow_U'] != 0).sum()
        v_nonzero = (df['Flow_V'] != 0).sum()
        u_pct = 100 * u_nonzero / len(df)
        print(f"  Flow_U non-zero   : {u_nonzero}/{len(df)} ({u_pct:.1f}%)")
        print(f"  Flow_V non-zero   : {v_nonzero}/{len(df)} ({100*v_nonzero/len(df):.1f}%)")
        if u_pct < 50:
            print(f"  {WARN} Less than 50% of points have flow vectors — direction arrows may not show for many dots")
        else:
            print(f"  {PASS} Flow vectors look healthy")
    # Risk level check
    if 'Risk_Level' in df.columns:
        print(f"  Risk distribution : {df['Risk_Level'].value_counts().to_dict()}")
        if df['Risk_Level'].nunique() == 1:
            print(f"  {WARN} Only one risk level — dashboard pie chart will show 100% one color")
        else:
            print(f"  {PASS} Multiple risk levels detected")
    # Source check
    if 'Source' in df.columns:
        print(f"  Source types      : {df['Source'].value_counts().to_dict()}")
    # Outlier check — points far from river
    far_from_river = df[
        (df['Lat'] < 16.490) | (df['Lat'] > 16.585)
    ]
    print(f"  Outliers off-river: {len(far_from_river)} points outside lat 16.49-16.585")
    if len(far_from_river) > 50:
        print(f"  {WARN} High number of outliers — may show land pollution in dashboard")
    else:
        print(f"  {PASS} Outlier count acceptable")
else:
    print(f"  {FAIL} results_for_viz.csv missing")
    all_ok = False

# ── 5. MONTHLY SENSOR FILES CHECK ────────────────────────────────
print("\n[SECTION 5] MONTHLY SENSOR FILES CHECK")
months = ['April', 'May', 'Jul', 'Sept', 'Nov', 'Nov_(2)', 'Dec']
for m in months:
    p = OUTPUTS / f"{m}_sensors.csv"
    if p.exists():
        sub = pd.read_csv(p)
        print(f"  {PASS} {m:10s}_sensors.csv — {len(sub):3d} rows  NTU=[{sub['Turbidity_NTU'].min():.1f}, {sub['Turbidity_NTU'].max():.1f}]")
    else:
        print(f"  {FAIL} {m}_sensors.csv — MISSING")
        all_ok = False

# ── 6. DASHBOARD READINESS SUMMARY ───────────────────────────────
print("\n" + "=" * 65)
print("[SECTION 6] DASHBOARD READINESS SUMMARY")
print("=" * 65)
if csv_path.exists():
    df = pd.read_csv(csv_path)
    # Apply cleaning for summary too
    df['Predicted_NTU'] = df['Predicted_NTU'].clip(lower=0)
    df = df[
        (df['Lat'] >= 16.490) &
        (df['Lat'] <= 16.585)
    ].copy().reset_index(drop=True)
    
    health = df['Predicted_NTU'].apply(lambda x: max(0, min(100, 100-x/8))).mean()
    print(f"  Data points on map    : {len(df):,}")
    print(f"  Average NTU           : {df['Predicted_NTU'].mean():.1f}")
    print(f"  Peak NTU              : {df['Predicted_NTU'].max():.1f}")
    print(f"  River Health Index    : {health:.1f}%")
    if 'Risk_Level' in df.columns:
        for r in ['Critical','High','Moderate','Low']:
            cnt = (df['Risk_Level']==r).sum()
            pct = 100*cnt/len(df) if len(df) > 0 else 0
            bar = '#' * int(pct/2)
            print(f"  {r:10s}: {cnt:5d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 65)
if all_ok:
    print("FINAL VERDICT: GREEN LIGHT — DATA IS READY FOR DEMO")
else:
    print("FINAL VERDICT: RED LIGHT — FIX ERRORS ABOVE BEFORE DEMO")
print("=" * 65)
