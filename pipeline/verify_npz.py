import numpy as np

d = np.load(
    r'C:\Users\karth\Downloads\New folder\outputs\final_fusion_dataset.npz',
    allow_pickle=True
)

print("Keys        :", list(d.keys()))
print("X_patches   :", d['X_patches'].shape,  d['X_patches'].dtype)
print("X_meta      :", d['X_meta'].shape,      d['X_meta'].dtype)
print("y_ntu       :", d['y_ntu'].shape,       d['y_ntu'].dtype)
print("y_label     :", d['y_label'].shape,     d['y_label'].dtype)
print("NTU range   :", round(float(d['y_ntu'].min()), 2),
      '-', round(float(d['y_ntu'].max()), 2))
print("NTU mean    :", round(float(d['y_ntu'].mean()), 2))

u, c = np.unique(d['y_label'], return_counts=True)
names = {0: 'Clean', 1: 'Moderate', 2: 'High'}
for lbl, cnt in zip(u, c):
    pct = 100 * cnt / len(d['y_label'])
    print(f"  label {lbl} ({names.get(int(lbl),'?'):8s}): {cnt:6d} patches ({pct:.1f}%)")

print("Sample info :", d['patch_info'][:3])
