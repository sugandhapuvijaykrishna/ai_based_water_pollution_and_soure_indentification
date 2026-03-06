"""
Landsat 8/9 Band Stacking and Feature Engineering
- Load SR bands (B3, B4, B5, B6)
- Calculate NDWI and NDTI indices
- Stack into feature tensor
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class LandsatProcessor:
    """Process Landsat 8/9 SR bands and compute spectral indices"""
    
    def __init__(self, band_dir: str):
        """
        Args:
            band_dir: Directory containing band GeoTIFF files
        """
        self.band_dir = Path(band_dir)
        self.profile = None
        
    def load_band(self, filename: str) -> np.ndarray:
        """
        Load single band as float32 array
        
        Args:
            filename: Band GeoTIFF filename
            
        Returns:
            Band array (H, W)
        """
        filepath = self.band_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Band file not found: {filepath}")
        
        with rasterio.open(filepath) as src:
            band = src.read(1).astype(np.float32)
            self.profile = src.profile  # Save for output
            
        print(f"✓ Loaded {filename}: shape {band.shape}")
        return band
    
    def stack_bands(self, band_files: dict) -> np.ndarray:
        """
        Stack multiple bands into single tensor
        
        Args:
            band_files: Dict mapping band names to filenames
                       e.g., {'B3': 'LC08_B3.tif', ...}
        
        Returns:
            Stacked array (H, W, num_bands)
        """
        bands = []
        for band_name in ['B3', 'B4', 'B5', 'B6']:
            band = self.load_band(band_files[band_name])
            bands.append(band)
        
        stacked = np.stack(bands, axis=-1)
        print(f"✓ Bands stacked: {stacked.shape} (H, W, Bands)")
        
        return stacked
    
    def normalize_band(self, band: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize band values
        
        Args:
            band: Input band array
            method: 'minmax' (0-1) or 'zscore' (standardize)
        
        Returns:
            Normalized band
        """
        if method == 'minmax':
            vmin, vmax = np.nanpercentile(band, [2, 98])
            band_norm = np.clip((band - vmin) / (vmax - vmin + 1e-6), 0, 1)
        elif method == 'zscore':
            band_norm = (band - np.nanmean(band)) / (np.nanstd(band) + 1e-6)
        else:
            raise ValueError(f"Unknown normalization: {method}")
        
        return band_norm.astype(np.float32)
    
    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index
        
        NDWI = (Green - NIR) / (Green + NIR)
        - Water: ~0 to 0.4
        - Land: negative values
        
        Args:
            green: SR_B3 (Green) band
            nir: SR_B5 (NIR) band
        
        Returns:
            NDWI array
        """
        ndwi = (green - nir) / (green + nir + 1e-6)
        return np.clip(ndwi, -1, 1).astype(np.float32)
    
    @staticmethod
    def calculate_ndti(red: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Tillage Index
        Useful for detecting suspended sediment/pollution
        
        NDTI = (Red - SWIR1) / (Red + SWIR1)
        
        Args:
            red: SR_B4 (Red) band
            swir1: SR_B6 (SWIR1) band
        
        Returns:
            NDTI array
        """
        ndti = (red - swir1) / (red + swir1 + 1e-6)
        return np.clip(ndti, -1, 1).astype(np.float32)
    
    def create_feature_stack(self, 
                            band_files: dict,
                            normalize: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Create full feature stack: 4 original bands + NDWI + NDTI
        
        Args:
            band_files: Band filenames dict
            normalize: Whether to normalize bands to [0,1]
        
        Returns:
            Feature stack (H, W, 6), metadata dict
        """
        # Load and stack bands
        stacked = self.stack_bands(band_files)
        
        # Extract individual bands
        green = stacked[:, :, 0]
        red = stacked[:, :, 1]
        nir = stacked[:, :, 2]
        swir1 = stacked[:, :, 3]
        
        # Normalize if requested
        if normalize:
            green = self.normalize_band(green)
            red = self.normalize_band(red)
            nir = self.normalize_band(nir)
            swir1 = self.normalize_band(swir1)
            stacked = np.stack([green, red, nir, swir1], axis=-1)
        
        # Calculate indices
        ndwi = self.calculate_ndwi(green, nir)
        ndti = self.calculate_ndti(red, swir1)
        
        # Stack all features
        features = np.concatenate(
            [stacked, ndwi[..., None], ndti[..., None]], 
            axis=-1
        )
        
        print(f"\n✓ Feature stack created: {features.shape}")
        print("  Channels: [B3(Green), B4(Red), B5(NIR), B6(SWIR1), NDWI, NDTI]")
        
        # Metadata
        metadata = {
            'shape': features.shape,
            'channels': ['B3_Green', 'B4_Red', 'B5_NIR', 'B6_SWIR1', 'NDWI', 'NDTI'],
            'profile': self.profile,
            'ndwi_range': (np.nanmin(ndwi), np.nanmax(ndwi)),
            'ndti_range': (np.nanmin(ndti), np.nanmax(ndti))
        }
        
        return features, metadata

if __name__ == "__main__":
    # Example usage
    processor = LandsatProcessor('./landsat_bands')
    
    band_files = {
        'B3': 'LC08_L2SR_T42QQK_20230101_B3.tif',
        'B4': 'LC08_L2SR_T42QQK_20230101_B4.tif',
        'B5': 'LC08_L2SR_T42QQK_20230101_B5.tif',
        'B6': 'LC08_L2SR_T42QQK_20230101_B6.tif'
    }
    
    features, metadata = processor.create_feature_stack(band_files, normalize=True)
    print("\nMetadata:", metadata)