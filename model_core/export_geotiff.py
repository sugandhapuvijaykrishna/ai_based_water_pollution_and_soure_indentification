"""
Export pollution classification results as GeoTIFF
Maintains geospatial metadata
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from pathlib import Path
from typing import Tuple, List
import json

class GeoTIFFExporter:
    """Export predictions as georeferenced GeoTIFF"""
    
    def __init__(self, reference_file: str, 
                 output_dir: str = './outputs'):
        """
        Args:
            reference_file: Reference GeoTIFF for geospatial metadata
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load reference profile
        with rasterio.open(reference_file) as src:
            self.profile = src.profile.copy()
            self.transform = src.transform
            self.crs = src.crs
            self.reference_shape = src.shape
        
        print(f"✓ Reference metadata loaded from {reference_file}")
        print(f"  CRS: {self.crs}")
        print(f"  Transform: {self.transform}")
        print(f"  Shape: {self.reference_shape}")
    
    def create_classification_map(self,
                                 predictions: np.ndarray,
                                 patch_coords: List[Tuple],
                                 patch_size: int = 32) -> np.ndarray:
        """
        Reconstruct full classification map from patch predictions
        
        Args:
            predictions: Patch class predictions (num_patches,)
            patch_coords: Patch coordinates list [(row, col), ...]
            patch_size: Patch size in pixels
        
        Returns:
            Full classification map (H, W)
        """
        # Initialize output map
        output_map = np.full(self.reference_shape, -1, dtype=np.int32)
        count_map = np.zeros(self.reference_shape, dtype=np.uint8)
        
        # Accumulate predictions
        for idx, (row, col) in enumerate(patch_coords):
            pred = predictions[idx]
            
            # Ensure patch fits within bounds
            end_row = min(row + patch_size, self.reference_shape[0])
            end_col = min(col + patch_size, self.reference_shape[1])
            
            patch_h = end_row - row
            patch_w = end_col - col
            
            # Write prediction (with overlap handling: average)
            output_map[row:end_row, col:end_col] = pred
            count_map[row:end_row, col:end_col] += 1
        
        # Fill unclassified areas with -1 (no data)
        unclassified = count_map == 0
        output_map[unclassified] = -1
        
        print(f"✓ Classification map created: {output_map.shape}")
        print(f"  Classified pixels: {(output_map >= 0).sum():,}")
        print(f"  Unclassified pixels: {(output_map < 0).sum():,}")
        
        return output_map
    
    def save_classification_geotiff(self,
                                   classification_map: np.ndarray,
                                   filename: str = 'pollution_classification.tif',
                                   class_colors: dict = None) -> str:
        """
        Save classification map as GeoTIFF
        
        Args:
            classification_map: Classification map array
            filename: Output filename
            class_colors: Optional dict for color interpretation
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Update profile for single-band output
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.int32,
            count=1,
            nodata=-1
        )
        
        # Write GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(classification_map, 1)
        
        print(f"✓ Classification saved: {output_path}")
        
        return str(output_path)
    
    def save_confidence_geotiff(self,
                               confidence_map: np.ndarray,
                               filename: str = 'pollution_confidence.tif') -> str:
        """
        Save prediction confidence as GeoTIFF
        
        Args:
            confidence_map: Confidence map (H, W) with values 0-1
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Update profile
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan
        )
        
        # Write GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(confidence_map, 1)
        
        print(f"✓ Confidence map saved: {output_path}")
        
        return str(output_path)
    
    def save_probability_geotiffs(self,
                                 probabilities: np.ndarray,
                                 patch_coords: List[Tuple],
                                 patch_size: int = 32,
                                 class_names: list = None) -> dict:
        """
        Save per-class probability maps as separate GeoTIFFs
        
        Args:
            probabilities: Prediction probabilities (num_patches, num_classes)
            patch_coords: Patch coordinates
            patch_size: Patch size
            class_names: Class names for filenames
        
        Returns:
            Dictionary with paths to saved files
        """
        class_names = class_names or ['clean', 'moderate', 'high']
        saved_files = {}
        
        num_classes = probabilities.shape[1]
        
        for class_idx in range(num_classes):
            # Create probability map
            prob_map = np.zeros(self.reference_shape, dtype=np.float32)
            
            for patch_idx, (row, col) in enumerate(patch_coords):
                prob = probabilities[patch_idx, class_idx]
                
                end_row = min(row + patch_size, self.reference_shape[0])
                end_col = min(col + patch_size, self.reference_shape[1])
                
                prob_map[row:end_row, col:end_col] = prob
            
            # Save
            filename = f'pollution_{class_names[class_idx]}_probability.tif'
            path = self.save_probability_geotiff(prob_map, filename)
            saved_files[class_names[class_idx]] = path
        
        return saved_files
    
    def save_probability_geotiff(self,
                                prob_map: np.ndarray,
                                filename: str) -> str:
        """Save single probability map"""
        output_path = self.output_dir / filename
        
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=np.nan
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prob_map, 1)
        
        print(f"✓ Probability map saved: {output_path}")
        return str(output_path)
    
    def create_rgb_composite(self,
                            classification_map: np.ndarray,
                            filename: str = 'pollution_rgb.tif') -> str:
        """
        Create RGB composite from classification
        Green=Clean, Yellow=Moderate, Red=High
        
        Args:
            classification_map: Classification map
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        rgb = np.zeros((*self.reference_shape, 3), dtype=np.uint8)
        
        # Color mapping
        rgb[classification_map == 0] = [0, 255, 0]      # Clean: Green
        rgb[classification_map == 1] = [255, 255, 0]    # Moderate: Yellow
        rgb[classification_map == 2] = [255, 0, 0]      # High: Red
        rgb[classification_map < 0] = [128, 128, 128]   # No data: Gray
        
        output_path = self.output_dir / filename
        
        profile = self.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=3
        )
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(3):
                dst.write(rgb[..., i], i + 1)
        
        print(f"✓ RGB composite saved: {output_path}")
        return str(output_path)
    
    def save_metadata(self, 
                     metadata: dict,
                     filename: str = 'results_metadata.json'):
        """Save metadata as JSON"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Metadata saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    exporter = GeoTIFFExporter('reference_band.tif')
    
    # Mock prediction
    predictions = np.random.randint(0, 3, 100)
    coords = [(i*16, j*16) for i in range(10) for j in range(10)]
    
    class_map = exporter.create_classification_map(predictions, coords)
    exporter.save_classification_geotiff(class_map)
    exporter.create_rgb_composite(class_map)