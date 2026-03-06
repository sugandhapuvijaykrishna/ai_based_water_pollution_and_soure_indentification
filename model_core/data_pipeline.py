"""
Complete data pipeline: loading, normalization, splitting
Memory-efficient for 16GB RAM

Extended with load_fusion_npz() for dual-branch regression pipeline.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from pathlib import Path
from typing import Tuple

class DataPipeline:
    """Handle data loading, normalization, and splitting"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
    
    def normalize_patches(self, patches: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize patch data
        
        Args:
            patches: Patch array (N, H, W, C) or (N, H, W)
            method: 'minmax' or 'zscore'
        
        Returns:
            Normalized patches
        """
        original_shape = patches.shape
        
        # Reshape to 2D for normalization
        N = patches.shape[0]
        flat_patches = patches.reshape(N, -1)
        
        if method == 'minmax':
            vmin = np.percentile(flat_patches, 2, axis=0)
            vmax = np.percentile(flat_patches, 98, axis=0)
            normalized = (flat_patches - vmin) / (vmax - vmin + 1e-6)
            normalized = np.clip(normalized, 0, 1)
        
        elif method == 'zscore':
            mean = np.mean(flat_patches, axis=0)
            std = np.std(flat_patches, axis=0)
            normalized = (flat_patches - mean) / (std + 1e-6)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reshape back
        normalized = normalized.reshape(original_shape)
        
        print(f"✓ Patches normalized using {method}")
        print(f"  Range: [{np.min(normalized):.4f}, {np.max(normalized):.4f}]")
        
        return normalized.astype(np.float32)
    
    def train_test_split(self,
                        patches: np.ndarray,
                        labels: np.ndarray,
                        coords: list = None,
                        test_size: float = 0.2,
                        val_split: float = 0.1) -> dict:
        """
        Stratified train/validation/test split
        
        Args:
            patches: Patch array
            labels: Labels array
            coords: Optional patch coordinates
            test_size: Fraction for test set
            val_split: Fraction of training for validation (handled by model)
        
        Returns:
            Dictionary with train/test data
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, coords_temp, coords_test = train_test_split(
            patches, labels, coords or list(range(len(labels))),
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Second split: train vs val (from train+val)
        X_train, X_val, y_train, y_val, coords_train, coords_val = train_test_split(
            X_temp, y_temp, coords_temp,
            test_size=val_split / (1 - test_size),
            random_state=self.random_state,
            stratify=y_temp
        )
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'coords_train': coords_train,
            'coords_val': coords_val,
            'coords_test': coords_test
        }
        
        # Print statistics
        self._print_split_stats(data)
        
        return data
    
    @staticmethod
    def _print_split_stats(data: dict):
        """Print train/val/test split statistics"""
        print("\n" + "="*60)
        print("DATA SPLIT STATISTICS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            y = data[f'y_{split}']
            X = data[f'X_{split}']
            unique, counts = np.unique(y, return_counts=True)
            
            label_names = {0: 'Clean', 1: 'Moderate', 2: 'High'}
            
            print(f"\n{split.upper()} SET:")
            print(f"  Samples: {len(y)}")
            print(f"  Shape: {X.shape}")
            
            for u, c in zip(unique, counts):
                pct = c / len(y) * 100
                print(f"    {label_names[u]:12s}: {c:4d} ({pct:5.1f}%)")
        
        print("="*60 + "\n")
    
    def save_dataset(self, data: dict, filepath: str):
        """Save processed dataset to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Dataset saved to {filepath}")
    
    @staticmethod
    def load_dataset(filepath: str) -> dict:
        """Load saved dataset"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Dataset loaded from {filepath}")
        return data

    # ─────────────────────────────────────────────────────────────────────
    # Fusion NPZ loader  (dual-branch regression)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_fusion_npz(npz_path: str,
                        val_size:  float = 0.15,
                        test_size: float = 0.15,
                        random_state: int = 42) -> dict:
        """
        Load the fusion dataset and split into train / val / test sets.

        The split is stratified by NTU quartile so that every split has a
        representative spread of low / medium / high turbidity values.

        Args:
            npz_path     : path to final_fusion_dataset.npz
            val_size     : fraction reserved for validation
            test_size    : fraction reserved for testing
            random_state : RNG seed

        Returns
        -------
        dict with keys::

            X_patches_train/val/test  (N, 32, 32, 6) float32
            X_meta_train/val/test     (N, 2)          float32  [lat, lon]
            y_ntu_train/val/test      (N,)            float32
            meta_scaler               fitted MinMaxScaler for (lat, lon)
        """
        npz = np.load(npz_path, allow_pickle=True)
        X_patches = npz['X_patches'].astype(np.float32)   # (N,32,32,6)
        X_meta    = npz['X_meta'].astype(np.float32)       # (N,2)
        y_ntu     = npz['y_ntu'].astype(np.float32)        # (N,)

        N = len(y_ntu)
        print(f'\n=== Fusion Dataset ===')
        print(f'  Total samples  : {N}')
        print(f'  X_patches      : {X_patches.shape}')
        print(f'  X_meta         : {X_meta.shape}')
        print(f'  y_ntu range    : [{y_ntu.min():.1f}, {y_ntu.max():.1f}]  '
              f'mean={y_ntu.mean():.1f}')

        # Stratification bins (quartiles of NTU)
        strat = np.digitize(y_ntu,
                            bins=np.percentile(y_ntu, [25, 50, 75]))

        # ── Split 1: (train+val) vs test ──────────────────────────────────
        idx = np.arange(N)
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_size,
            stratify=strat, random_state=random_state
        )

        # ── Split 2: train vs val ─────────────────────────────────────────
        strat_tv = strat[idx_tv]
        val_frac  = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac,
            stratify=strat_tv, random_state=random_state
        )

        # ── Normalise GPS coordinates (lat, lon) → [0, 1] ─────────────────
        meta_scaler = MinMaxScaler()
        X_meta_scaled = meta_scaler.fit_transform(X_meta)

        def _s(arr, idx): return arr[idx]

        data = {
            # patches (already float32, normalised during fusion build)
            'X_patches_train': _s(X_patches, idx_train),
            'X_patches_val':   _s(X_patches, idx_val),
            'X_patches_test':  _s(X_patches, idx_test),
            # GPS metadata (scaled)
            'X_meta_train': _s(X_meta_scaled, idx_train).astype(np.float32),
            'X_meta_val':   _s(X_meta_scaled, idx_val).astype(np.float32),
            'X_meta_test':  _s(X_meta_scaled, idx_test).astype(np.float32),
            # NTU targets
            'y_ntu_train': _s(y_ntu, idx_train),
            'y_ntu_val':   _s(y_ntu, idx_val),
            'y_ntu_test':  _s(y_ntu, idx_test),
            # raw meta for heatmap (unscaled)
            'X_meta_raw_test': _s(X_meta, idx_test),
            'meta_scaler':  meta_scaler,
        }

        print(f'  Train : {len(idx_train)}  '
              f'Val : {len(idx_val)}  '
              f'Test : {len(idx_test)}')
        print('======================\n')
        return data


if __name__ == '__main__':
    # Classification example
    patches = np.random.randn(1000, 32, 32, 6).astype(np.float32)
    labels  = np.random.randint(0, 3, 1000)
    pipeline = DataPipeline()
    patches_norm = pipeline.normalize_patches(patches)
    data = pipeline.train_test_split(patches_norm, labels)

    # Regression / fusion example
    # data_fusion = DataPipeline.load_fusion_npz('../outputs/final_fusion_dataset.npz')