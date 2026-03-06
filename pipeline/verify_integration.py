"""
Final Integration Audit
========================
Verifies all files are present and correct before demo day.
Run this from the project root directory.
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Relative paths - works on any machine
ROOT_DIR    = Path(__file__).parent
OUTPUTS_DIR = ROOT_DIR.parent / "outputs"
DATA_PATH   = OUTPUTS_DIR / "final_fusion_dataset.npz"
MODEL_H5    = OUTPUTS_DIR / "best_fusion_model.h5"
MODEL_DIR   = OUTPUTS_DIR / "best_fusion_model"
RESULTS_CSV = OUTPUTS_DIR / "results_for_viz.csv"

RIVER_DATA    = OUTPUTS_DIR / "river_dataset.npz"
RIVER_MODEL_H5 = OUTPUTS_DIR / "river_model.h5"
RIVER_MODEL_DIR = OUTPUTS_DIR / "river_model"
RIVER_METRICS  = OUTPUTS_DIR / "river_metrics.txt"
RIVER_CSV      = OUTPUTS_DIR / "results_for_viz.csv"


def verify_integration():
    print("=" * 60)
    print("RIVER POLLUTION PROJECT: FINAL INTEGRATION AUDIT")
    print("=" * 60)

    all_passed = True

    # Task 1: File Presence
    print("\n[TASK 1] FILE PRESENCE CHECK")
    files_to_check = {
        "Data Source (.npz)":  DATA_PATH,
        "Model (.h5)":         MODEL_H5,
        "Viz Handoff (.csv)":  RESULTS_CSV,
    }
    for name, path in files_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"  {name:25s}: OK  ({size_mb:.2f} MB)")
        else:
            print(f"  {name:25s}: MISSING - {path}")
            all_passed = False

    # Also check SavedModel format
    if MODEL_DIR.exists():
        print(f"  {'SavedModel (folder)':25s}: OK")
    else:
        print(f"  {'SavedModel (folder)':25s}: Not found (h5 is sufficient)")

    # Task 2: Data Integrity
    print("\n[TASK 2] DATA INTEGRITY AUDIT")
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)

        ntu_min = df['Predicted_NTU'].min()
        ntu_max = df['Predicted_NTU'].max()
        ntu_std = df['Predicted_NTU'].std()
        print(f"  NTU Range         : [{ntu_min:.2f} - {ntu_max:.2f}]  Std: {ntu_std:.2f}")

        if ntu_std < 1.0:
            print("  WARNING: NTU has near-zero variance. Model may not have converged.")
            all_passed = False
        else:
            print("  NTU Variance      : OK")

        lat_min, lat_max = df['Lat'].min(), df['Lat'].max()
        lon_min, lon_max = df['Lon'].min(), df['Lon'].max()
        print(f"  Lat Bounds        : [{lat_min:.4f} - {lat_max:.4f}]")
        print(f"  Lon Bounds        : [{lon_min:.4f} - {lon_max:.4f}]")

        # Check coordinates are in Krishna River area
        in_range = (16.4 <= lat_min <= 16.6 and 16.4 <= lat_max <= 16.6 and
                    80.3 <= lon_min <= 81.0 and 80.3 <= lon_max <= 81.0)
        print(f"  Coordinate Check  : {'OK - Krishna River area' if in_range else 'WARNING - Outside expected range'}")

        if 'Flow_U' in df.columns:
            u_nonzero = (df['Flow_U'] != 0).sum()
            v_nonzero = (df['Flow_V'] != 0).sum()
            print(f"  Flow Vectors      : U={u_nonzero}/{len(df)}  V={v_nonzero}/{len(df)}")

        if 'Risk_Level' in df.columns:
            print(f"  Risk Levels       : {df['Risk_Level'].value_counts().to_dict()}")

        print(f"  Total Records     : {len(df):,}")
    else:
        print("  SKIPPED: results_for_viz.csv missing")
        all_passed = False

    # Task 3: Model Architecture
    print("\n[TASK 3] MODEL ARCHITECTURE CHECK")
    model_found = MODEL_H5.exists() or MODEL_DIR.exists()

    if model_found:
        try:
            if MODEL_H5.exists():
                model = tf.keras.models.load_model(str(MODEL_H5), compile=False)
            else:
                model = tf.keras.models.load_model(str(MODEL_DIR), compile=False)

            print(f"  Model loaded      : OK")
            print(f"  Input names       : {model.input_names}")

            # Check shapes
            for inp in model.inputs:
                shape = tuple(inp.shape)
                print(f"  Input '{inp.name}': {shape}")

            print(f"  Output shape      : {tuple(model.output.shape)}")
            print(f"  Total params      : {model.count_params():,}")
            print("  Architecture      : OK")

        except Exception as e:
            print(f"  ERROR loading model: {type(e).__name__}: {e}")
            print("  TIP: Try re-running trainer_fusion.py to regenerate the model.")
            all_passed = False
    else:
        print("  SKIPPED: No model file found. Run trainer_fusion.py first.")
        all_passed = False

    # Task 4: Dashboard Check
    print("\n[TASK 4] DASHBOARD FILE CHECK")
    dashboard = ROOT_DIR.parent / "dashboard" / "command_center.py"
    if dashboard.exists():
        print(f"  command_center.py : OK")
        print(f"  Run with          : streamlit run command_center.py")
    else:
        print(f"  command_center.py : MISSING")
        all_passed = False

    # Task 5: River Pipeline Check
    print("\n[TASK 5] RIVER PIPELINE CHECK")
    river_files = {
        "River Dataset (.npz)":  RIVER_DATA,
        "River Model (.h5)":     RIVER_MODEL_H5,
        "River Metrics (.txt)":  RIVER_METRICS,
        "River CSV (.csv)":      RIVER_CSV,
    }
    for name, path in river_files.items():
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"  {name:25s}: OK  ({size_mb:.2f} MB)")
        else:
            print(f"  {name:25s}: MISSING")
            all_passed = False

    # Final verdict
    print("\n" + "=" * 60)
    if all_passed:
        print("FINAL VERDICT: GREEN LIGHT")
        print("SYSTEM IS READY FOR PRESENTATION.")
    else:
        print("FINAL VERDICT: RED LIGHT")
        print("ACTION REQUIRED: Fix the errors above before demo.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    verify_integration()

