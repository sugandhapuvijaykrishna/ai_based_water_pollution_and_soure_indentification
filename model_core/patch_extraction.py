"""
Spatial patch extraction with overlap
Efficient memory-safe extraction for CNN training
"""

import numpy as np
import rasterio
from typing import Tuple, List
from pathlib import Path

class PatchExtractor:
    """Extract overlapping patches from feature maps and labels"""
    
    def __init__(self, patch_size: int = 32, overlap_ratio: float = 0.5):
        """
        Args:
            patch_size: Patch height and width (e.g., 32, 64)
            overlap_ratio: Overlap as fraction (0.5 = 50% overlap)
        """
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap_ratio))
        self.overlap_ratio = overlap_ratio
        
        print(f"✓ Patch Extractor initialized:")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Stride: {self.stride} (overlap: {overlap_ratio*100:.0f}%)")
    
    def extract_patches(self, 
                       img: np.ndarray, 
                       skip_nodata: bool = True,
                       mask: np.ndarray = None) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Extract patches from image with optional nodata handling
        
        Args:
            img: Input array (H, W) or (H, W, C)
            skip_nodata: Skip patches containing NaN or invalid values
            mask: Optional 2D boolean array (H, W). Patches with water
                  fraction below 0.5 are skipped.
        
        Returns:
            patches: Array (num_patches, patch_size, patch_size) or (num_patches, patch_size, patch_size, C)
            coords: List of (row, col) coordinates for each patch
        """
        if len(img.shape) == 2:
            h, w = img.shape
            channels = 1
            img_3d = img[..., None]
        else:
            h, w, channels = img.shape
            img_3d = img
        
        patches = []
        coords = []
        skipped = 0
        
        # Extract patches
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                # Water mask filter: skip patches with < 50% water
                if mask is not None:
                    patch_mask = mask[i:i+self.patch_size, j:j+self.patch_size]
                    if patch_mask.mean() < 0.65:
                        skipped += 1
                        continue

                patch = img_3d[i:i+self.patch_size, j:j+self.patch_size, :]
                
                # Check if patch is valid
                if skip_nodata and (np.isnan(patch).any() or np.isinf(patch).any()):
                    skipped += 1
                    continue
                
                patches.append(patch)
                coords.append((i, j))
        
        patches_arr = np.array(patches)
        
        # Remove channel dim if input was 2D
        if channels == 1:
            patches_arr = patches_arr.squeeze(axis=-1)
        
        print(f"✓ Extracted {len(patches)} patches from ({h}, {w})")
        if skipped > 0:
            print(f"  (Skipped {skipped} invalid patches)")
        
        return patches_arr, coords
    
    def load_labels_and_extract(self, 
                               label_file: str,
                               patch_coords: List[Tuple],
                               method: str = 'majority') -> np.ndarray:
        """
        Load label raster and assign labels to patches
        
        Args:
            label_file: Path to label GeoTIFF (values: 0, 1, 2)
            patch_coords: List of (row, col) patch coordinates
            method: 'majority' (majority vote) or 'center' (center pixel)
        
        Returns:
            labels: Array of class labels for each patch
        """
        with rasterio.open(label_file) as src:
            label_map = src.read(1).astype(np.int32)
        
        labels = []
        invalid = 0
        
        for i, (row, col) in enumerate(patch_coords):
            patch_label = label_map[row:row+self.patch_size, col:col+self.patch_size]
            
            if method == 'majority':
                # Use majority class in patch
                vals, counts = np.unique(patch_label[patch_label >= 0], return_counts=True)
                if len(vals) == 0:
                    invalid += 1
                    continue
                main_label = vals[np.argmax(counts)]
            elif method == 'center':
                # Use center pixel
                cy, cx = self.patch_size // 2
                main_label = patch_label[cy, cx]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Validate label
            if main_label < 0 or main_label > 2:
                invalid += 1
                continue
            
            labels.append(main_label)
        
        labels_arr = np.array(labels)
        
        print(f"✓ Assigned labels to {len(labels)} patches")
        print(f"  Class distribution: {np.bincount(labels_arr)}")
        if invalid > 0:
            print(f"  (Skipped {invalid} patches with invalid labels)")
        
        return labels_arr
    
    def get_statistics(self, patches: np.ndarray, labels: np.ndarray = None):
        """Print patch statistics"""
        print("\n" + "="*60)
        print("PATCH STATISTICS")
        print("="*60)
        print(f"Total patches: {len(patches)}")
        print(f"Patch shape: {patches[0].shape}")
        print(f"Data type: {patches.dtype}")
        print(f"Value range: [{np.min(patches):.4f}, {np.max(patches):.4f}]")
        print(f"Mean: {np.mean(patches):.4f}, Std: {np.std(patches):.4f}")
        
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nLabel distribution:")
            label_names = {0: 'Clean Water', 1: 'Moderate Pollution', 2: 'High Pollution'}
            for u, c in zip(unique, counts):
                pct = c / len(labels) * 100
                print(f"  {label_names[u]}: {c} ({pct:.1f}%)")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    # Example usage
    extractor = PatchExtractor(patch_size=32, overlap_ratio=0.5)
    
    # Mock data
    features = np.random.randn(512, 512, 6).astype(np.float32)
    
    patches, coords = extractor.extract_patches(features)
    extractor.get_statistics(patches)