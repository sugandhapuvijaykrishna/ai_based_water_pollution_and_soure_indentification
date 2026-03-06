import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

csv_path = r"C:\Users\karth\Downloads\New folder\outputs\results_for_viz.csv"
model_path = r"C:\Users\karth\Downloads\New folder\outputs\best_fusion_model.h5"

print("="*60)
print("RIVER POLLUTION PROJECT FINAL AUDIT")
print("="*60)

# --- TASK 2: DATA INTEGRITY ---
print("\n[TASK 2] DATA INTEGRITY AUDIT")
if Path(csv_path).exists():
    df = pd.read_csv(csv_path)
    print("✓ CSV Found.")
    stats = df['Predicted_NTU'].describe()
    print(f"NTU Stats: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Std={stats['std']:.2f}")
    
    lat_min, lat_max = df['Lat'].min(), df['Lat'].max()
    lon_min, lon_max = df['Lon'].min(), df['Lon'].max()
    print(f"Coordinates: Lat({lat_min:.4f} to {lat_max:.4f}), Lon({lon_min:.4f} to {lon_max:.4f})")
    
    u_nonzero = (df['Flow_U'] != 0).sum()
    v_nonzero = (df['Flow_V'] != 0).sum()
    print(f"Non-zero Flow Vectors: U({u_nonzero}), V({v_nonzero})")
else:
    print("✗ ERROR: results_for_viz.csv NOT FOUND")

# --- TASK 3: ARCHITECTURE ---
print("\n[TASK 3] ARCHITECTURE CONFIRMATION")
if Path(model_path).exists():
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model Weights Loaded.")
        
        print("\nInput layers (Shapes):")
        for inp in model.inputs:
            print(f"  - {inp.name}: {inp.shape}")
            
        # Verify 32x32x6 and (2,)
        shapes = [tuple(inp.shape.as_list()) for inp in model.inputs]
        has_patch = any(s == (None, 32, 32, 6) for s in shapes)
        has_meta = any(s == (None, 2) for s in shapes)
        
        if has_patch and has_meta:
            print("\n✓ Architecture Check: Passed.")
        else:
            print(f"\n✗ Architecture Check: Failed. Detected shapes: {shapes}")
            
    except Exception as e:
        print(f"✗ ERROR Loading Model: {type(e).__name__}: {e}")
else:
    print("✗ ERROR: best_fusion_model.h5 NOT FOUND")

print("\n" + "="*60)
print("AUDIT COMPLETE.")
print("="*60)
