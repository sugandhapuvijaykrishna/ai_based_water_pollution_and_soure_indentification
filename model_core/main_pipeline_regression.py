"""
Master Pipeline: End-to-End Regression Model
Water Pollution Index Prediction (Continuous: 0-100)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from pathlib import Path

# Import modules
from gpu_setup import configure_gpu, get_device_info
from preprocessing import LandsatProcessor
from patch_extraction import PatchExtractor
from data_pipeline import DataPipeline
from model_regression import PollutionCNNRegression, RegressionMetrics
from training_regression import RegressionTrainingConfig, RegressionTrainer, RegressionEvaluator
from evaluation_regression import RegressionVisualization
from export_geotiff import GeoTIFFExporter

# Configuration
BAND_DIR = './landsat_data'
LABEL_FILE = './labels/pollution_index.tif'  # Continuous values 0-1
REFERENCE_BAND = './landsat_data/LC08_L2SR_B3.tif'
OUTPUT_DIR = './results_regression'

PATCH_SIZE = 32
OVERLAP_RATIO = 0.5


def main():
    print("\n" + "="*70)
    print("WATER POLLUTION DETECTION - REGRESSION PIPELINE")
    print("Continuous Pollution Index Prediction (0-100)")
    print("Krishna River, Vijayawada, India")
    print("="*70)
    
    # ===== STEP 1: GPU SETUP =====
    print("\n[1/9] CONFIGURING GPU...")
    configure_gpu(gpu_memory_limit_mb=12000)
    get_device_info()
    
    # ===== STEP 2: PREPROCESS BANDS =====
    print("\n[2/9] PREPROCESSING LANDSAT BANDS...")
    processor = LandsatProcessor(BAND_DIR)
    
    band_files = {
        'B3': 'LC08_L2SR_B3.tif',
        'B4': 'LC08_L2SR_B4.tif',
        'B5': 'LC08_L2SR_B5.tif',
        'B6': 'LC08_L2SR_B6.tif'
    }
    
    features, metadata = processor.create_feature_stack(band_files, normalize=True)
    print(f"✓ Features shape: {features.shape}")
    
    # ===== STEP 3: EXTRACT PATCHES =====
    print("\n[3/9] EXTRACTING SPATIAL PATCHES...")
    extractor = PatchExtractor(patch_size=PATCH_SIZE, overlap_ratio=OVERLAP_RATIO)
    patches, coords = extractor.extract_patches(features)
    
    # Load continuous pollution index labels
    labels = extractor.load_labels_and_extract(LABEL_FILE, coords, method='mean')
    # Normalize labels to [0, 1] if needed
    labels = np.clip(labels, 0, 1).astype(np.float32)
    
    extractor.get_statistics(patches, labels)
    
    # ===== STEP 4: PREPARE DATA =====
    print("\n[4/9] PREPARING TRAINING DATA...")
    pipeline = DataPipeline()
    patches_norm = pipeline.normalize_patches(patches, method='minmax')
    
    # For regression, use continuous split
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test, coords_temp, coords_test = train_test_split(
        patches_norm, labels, coords,
        test_size=0.15,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val, coords_train, coords_val = train_test_split(
        X_temp, y_temp, coords_temp,
        test_size=0.15 / 0.85,
        random_state=42
    )
    
    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Target range: [{np.min(labels):.4f}, {np.max(labels):.4f}]")
    
    # ===== STEP 5: BUILD REGRESSION MODEL =====
    print("\n[5/9] BUILDING REGRESSION CNN MODEL...")
    model_builder = PollutionCNNRegression(
        input_shape=(PATCH_SIZE, PATCH_SIZE, 6),
        learning_rate=0.001,
        l2_strength=1e-4,
        dropout_rate=0.5
    )
    
    model = model_builder.build_model()
    model = model_builder.compile_model(loss_function='mse')
    model_builder.print_model_summary(verbose=1)
    
    # ===== STEP 6: TRAIN MODEL =====
    print("\n[6/9] TRAINING REGRESSION MODEL...")
    config = RegressionTrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        early_stop_patience=15,
        reduce_lr_patience=7,
        model_dir=OUTPUT_DIR,
        log_dir=f'{OUTPUT_DIR}/logs'
    )
    
    trainer = RegressionTrainer(config)
    train_info = trainer.train(
        model, X_train, y_train, X_val, y_val,
        data_augmentation=False
    )
    
    # ===== STEP 7: EVALUATE =====
    print("\n[7/9] EVALUATING REGRESSION MODEL...")
    y_test_pred_raw = model.predict(X_test, batch_size=32, verbose=0)
    y_test_pred = y_test_pred_raw.flatten()
    
    evaluator = RegressionEvaluator()
    metrics = evaluator.print_report(y_test, y_test_pred)
    
    # ===== STEP 8: VISUALIZE RESULTS =====
    print("\n[8/9] GENERATING VISUALIZATIONS...")
    
    # Scale to 0-100 for visualization
    y_test_index = y_test * 100
    y_test_pred_index = y_test_pred * 100
    
    viz = RegressionVisualization(output_dir=OUTPUT_DIR)
    viz.plot_training_history(train_info['history'])
    viz.plot_predictions_distribution(y_test_index, y_test_pred_index)
    viz.plot_error_analysis(y_test_index, y_test_pred_index)
    viz.plot_quantile_regression(y_test_index, y_test_pred_index)
    
    # ===== STEP 9: EXPORT GEOTIFF =====
    print("\n[9/9] EXPORTING RESULTS AS GEOTIFF...")
    exporter = GeoTIFFExporter(REFERENCE_BAND, output_dir=OUTPUT_DIR)
    
    # Full prediction
    all_pred_raw = model.predict(patches_norm, batch_size=32, verbose=0)
    all_pred = all_pred_raw.flatten()
    
    # Create continuous pollution index map
    pollution_index_map = exporter.create_classification_map(
        (all_pred * 100).astype(np.int32), coords, PATCH_SIZE
    )
    
    # Save as GeoTIFF
    exporter.save_classification_geotiff(
        pollution_index_map,
        filename='pollution_index_continuous.tif'
    )
    
    # Save as continuous float GeoTIFF
    exporter.save_confidence_geotiff(
        all_pred.reshape(-1, 1),  # Placeholder: should reshape properly
        filename='pollution_index_probability.tif'
    )
    
    # Create categorical map (clean/moderate/high based on thresholds)
    thresholds = [0.33, 0.67]  # 33, 67 on 0-100 scale
    categorical_map = np.digitize(all_pred, bins=thresholds)
    exporter.save_classification_geotiff(
        categorical_map,
        filename='pollution_category.tif'
    )
    
    # Summary
    summary = {
        'model': 'PollutionCNNRegression',
        'dataset': {
            'location': 'Krishna River, Vijayawada, India',
            'bands': metadata['channels'],
            'patch_size': PATCH_SIZE,
            'total_patches': len(patches),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'training': {
            'epochs