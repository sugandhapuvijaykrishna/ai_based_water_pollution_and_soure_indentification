"""
Master pipeline: End-to-end water pollution detection
Krishna River, Vijayawada, India

Modes
-----
  python main_pipeline.py                   # original classification pipeline
  python main_pipeline.py --mode fusion     # dual-branch NTU regression
  python main_pipeline.py --mode fusion --epochs 2   # quick smoke test
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe on servers)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import pearsonr

# Import modules
from gpu_setup import configure_gpu, get_device_info
from preprocessing import LandsatProcessor
from patch_extraction import PatchExtractor
from data_pipeline import DataPipeline
from model import PollutionCNN, DualBranchFusionModel
from training import TrainingConfig, train_model, train_fusion_model
from evaluation import ModelEvaluator
from export_geotiff import GeoTIFFExporter

# ── Shared paths ───────────────────────────────────────────────────────────────────
# Classification pipeline (legacy / Landsat)
BAND_DIR       = './landsat_data'
LABEL_FILE     = './labels/pollution_labels.tif'
REFERENCE_BAND = './landsat_data/LC08_L2SR_B3.tif'
OUTPUT_DIR     = './results'

# Fusion pipeline (Sentinel-2 + NTU sensor)
NPZ_PATH       = str(Path(__file__).parent.parent / 'outputs' / 'final_fusion_dataset.npz')

PATCH_SIZE = 32
OVERLAP_RATIO = 0.5

def main():
    print('\n' + '=' * 70)
    print('WATER POLLUTION DETECTION - COMPLETE PIPELINE')
    print('Krishna River, Vijayawada, India')
    print('=' * 70)

    # ===== STEP 1: GPU SETUP =====
    print('\n[1/8] CONFIGURING GPU...')
    configure_gpu(gpu_memory_limit_mb=12000)
    get_device_info()

    # ===== STEP 2: PREPROCESS BANDS =====
    print('\n[2/8] PREPROCESSING LANDSAT BANDS...')
    processor = LandsatProcessor(BAND_DIR)

    band_files = {
        'B3': 'LC08_L2SR_B3.tif',
        'B4': 'LC08_L2SR_B4.tif',
        'B5': 'LC08_L2SR_B5.tif',
        'B6': 'LC08_L2SR_B6.tif'
    }

    features, metadata = processor.create_feature_stack(band_files, normalize=True)
    print(f'Features shape: {features.shape}')

    # ===== STEP 3: EXTRACT PATCHES =====
    print('\n[3/8] EXTRACTING SPATIAL PATCHES...')
    extractor = PatchExtractor(patch_size=PATCH_SIZE, overlap_ratio=OVERLAP_RATIO)
    patches, coords = extractor.extract_patches(features)
    labels = extractor.load_labels_and_extract(LABEL_FILE, coords, method='majority')
    extractor.get_statistics(patches, labels)

    # ===== STEP 4: PREPARE DATA =====
    print('\n[4/8] PREPARING TRAINING DATA...')
    pipeline = DataPipeline()
    patches_norm = pipeline.normalize_patches(patches, method='minmax')
    data = pipeline.train_test_split(patches_norm, labels, test_size=0.2)

    # ===== STEP 5: BUILD MODEL =====
    print('\n[5/8] BUILDING CNN MODEL...')
    model = PollutionCNN.simple_cnn(input_shape=(PATCH_SIZE, PATCH_SIZE, 6))
    model = PollutionCNN.compile_model(model, learning_rate=0.001)
    PollutionCNN.print_model_info(model)

    # ===== STEP 6: TRAIN MODEL =====
    print('\n[6/8] TRAINING MODEL...')
    config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stop_patience=10,
        model_dir=OUTPUT_DIR
    )

    train_info = train_model(model, data, config, data_augmentation=False)

    # ===== STEP 7: EVALUATE =====
    print('\n[7/8] EVALUATING MODEL...')
    evaluator = ModelEvaluator()

    y_test_pred = model.predict(data['X_test'], batch_size=32)
    metrics = evaluator.print_report(data['y_test'], y_test_pred)

    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_training_history(train_info['history'])
    evaluator.plot_prediction_uncertainty(y_test_pred, data['y_test'])

    # ===== STEP 8: EXPORT GEOTIFF =====
    print('\n[8/8] EXPORTING RESULTS AS GEOTIFF...')
    exporter = GeoTIFFExporter(REFERENCE_BAND, output_dir=OUTPUT_DIR)

    all_predictions = model.predict(patches, batch_size=32)
    all_classes = np.argmax(all_predictions, axis=1)

    class_map = exporter.create_classification_map(all_classes, coords, PATCH_SIZE)
    exporter.save_classification_geotiff(class_map)
    exporter.create_rgb_composite(class_map)

    confidence = np.max(all_predictions, axis=1)
    confidence_map = exporter.create_classification_map(confidence, coords, PATCH_SIZE)
    exporter.save_confidence_geotiff(confidence_map)

    summary = {
        'dataset': {
            'location': 'Krishna River, Vijayawada, India',
            'bands': metadata['channels'],
            'total_patches': len(patches)
        },
        'training': {
            'epochs_trained': train_info['epochs_trained'],
            'batch_size': config.batch_size
        },
        'evaluation': {
            'test_accuracy': float(metrics['accuracy']),
            'test_precision': float(metrics['precision']),
            'test_recall': float(metrics['recall']),
            'test_f1': float(metrics['f1'])
        }
    }

    exporter.save_metadata(summary)

    print('\n' + '=' * 70)
    print('PIPELINE COMPLETE!')
    print('=' * 70)
    print(f'Results saved to: {OUTPUT_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
def main_fusion(epochs: int = 50,
               batch_size: int = 32,
               learning_rate: float = 5e-4,
               npz_path: str = None,
               output_dir: str = None,
               generate_heatmap: bool = True) -> None:
    """
    Dual-Branch Fusion Pipeline
    ===========================
    Loads final_fusion_dataset.npz, trains DualBranchFusionModel, evaluates
    on the test set (MAE / RMSE / R²), and saves the NTU directional heatmap.

    Args:
        epochs          : training epochs
        batch_size      : mini-batch size
        learning_rate   : Adam learning rate
        npz_path        : override default NPZ location
        output_dir      : override default results folder
        generate_heatmap: save the NTU gradient scatter-plot PNG
    """
    npz  = npz_path   or NPZ_PATH
    odir = Path(output_dir or OUTPUT_DIR)
    odir.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 70)
    print('DUAL-BRANCH FUSION PIPELINE  –  NTU Regression')
    print('Krishna River, Vijayawada, India')
    print('=' * 70)

    # ── 1. GPU ──────────────────────────────────────────────────────────────────
    print('\n[1/5] CONFIGURING GPU...')
    configure_gpu(gpu_memory_limit_mb=12000)
    get_device_info()

    # ── 2. Load & split NPZ ───────────────────────────────────────────────────
    print('\n[2/5] LOADING FUSION DATASET...')
    data = DataPipeline.load_fusion_npz(npz)

    # ── 3. Build & compile model ──────────────────────────────────────────────
    print('\n[3/5] BUILDING DUAL-BRANCH FUSION MODEL...')
    model = DualBranchFusionModel.build(
        patch_shape=(32, 32, 6),
        meta_dim=2,
    )
    model = DualBranchFusionModel.compile(model, learning_rate=learning_rate)
    DualBranchFusionModel.print_info(model)

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print('\n[4/5] TRAINING...')
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stop_patience=10,
        model_dir=str(odir),
        mode='regression',
    )

    # Unscaled val meta for directional gradient callback
    X_meta_raw_val = data.get('X_meta_raw_test')   # use test raw coords as proxy

    train_info = train_fusion_model(
        model, data, config,
        X_meta_raw_val=X_meta_raw_val
    )

    # ── 5. Evaluate on test set ──────────────────────────────────────────────
    print('\n[5/5] EVALUATING...')
    Xp_test = data['X_patches_test']
    Xm_test = data['X_meta_test']
    y_test  = data['y_ntu_test']

    y_pred = model.predict([Xp_test, Xm_test], batch_size=batch_size).flatten()

    mae  = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))

    # R² (coefficient of determination)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    print('\n' + '=' * 60)
    print('TEST SET REGRESSION METRICS')
    print('=' * 60)
    print(f'  MAE   : {mae:.2f} NTU')
    print(f'  RMSE  : {rmse:.2f} NTU')
    print(f'  R²    : {r2:.4f}')
    print('=' * 60)

    # ── 6. Directional NTU Heatmap ───────────────────────────────────────────
    if generate_heatmap:
        _save_ntu_heatmap(
            lats   = data['X_meta_raw_test'][:, 0],
            lons   = data['X_meta_raw_test'][:, 1],
            y_pred = y_pred,
            y_true = y_test,
            output_dir = odir,
            mae  = mae,
            rmse = rmse,
            r2   = r2,
        )

    print('\n' + '=' * 70)
    print('FUSION PIPELINE COMPLETE!')
    print('=' * 70)
    print(f'Epochs trained : {train_info["epochs_trained"]}')
    print(f'Outputs saved  : {odir}')


def _save_ntu_heatmap(lats, lons, y_pred, y_true, output_dir,
                      mae, rmse, r2):
    """
    Save a scatter-plot heatmap of predicted NTU values overlaid on
    geographic (lon, lat) coordinates.

    The colour gradient visualises the directional NTU decay:
      dark red / orange  → high turbidity (plume core, near source)
      blue / purple      → low turbidity (clean water, downstream)

    Two subplots are produced:
      Left  – Predicted NTU
      Right – Prediction Error  |predicted − true|
    """
    output_dir = Path(output_dir)
    heatmap_path = output_dir / 'ntu_directional_heatmap.png'

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')

    # ── Left: Predicted NTU heatmap ────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#0d1117')
    sc1 = ax1.scatter(
        lons, lats,
        c=y_pred,
        cmap='RdYlBu_r',          # Red=high NTU  Blue=low NTU
        s=18,
        alpha=0.8,
        edgecolors='none',
    )
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cbar1.set_label('Predicted Turbidity (NTU)', color='white', fontsize=10)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    # Annotate directional arrow (west → east = upstream → downstream)
    ax1.annotate(
        '', xy=(lons.max(), np.median(lats)),
        xytext=(lons.min(), np.median(lats)),
        arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
    )
    ax1.text(
        (lons.min() + lons.max()) / 2, np.median(lats) + 0.005,
        'River Flow Direction →',
        ha='center', color='white', fontsize=8
    )

    ax1.set_xlabel('Longitude', color='white', fontsize=10)
    ax1.set_ylabel('Latitude',  color='white', fontsize=10)
    ax1.set_title('Predicted NTU  –  Directional Heatmap',
                  color='white', fontsize=12, pad=10)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    # ── Right: Prediction error ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    error = np.abs(y_pred - y_true)
    sc2 = ax2.scatter(
        lons, lats,
        c=error,
        cmap='hot',
        s=18,
        alpha=0.8,
        edgecolors='none',
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label('|Predicted − True| NTU', color='white', fontsize=10)
    cbar2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

    ax2.set_xlabel('Longitude', color='white', fontsize=10)
    ax2.set_ylabel('Latitude',  color='white', fontsize=10)
    ax2.set_title('Absolute Prediction Error  –  Spatial Distribution',
                  color='white', fontsize=12, pad=10)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    # ── Shared metrics footer ────────────────────────────────────────────
    fig.suptitle(
        f'Krishna River  ∣  NTU Regression  ∣  '
        f'MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}',
        color='white', fontsize=13, y=1.01
    )

    plt.tight_layout()
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Heatmap saved → {heatmap_path}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Water Pollution Detection  –  Krishna River'
    )
    parser.add_argument(
        '--mode', choices=['classify', 'fusion'], default='classify',
        help='classify = Landsat CNN  |  fusion = Sentinel-2 dual-branch regression'
    )
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=5e-4,
                        help='Learning rate (fusion mode only)')
    parser.add_argument('--npz',        type=str,   default=None,
                        help='Override path to final_fusion_dataset.npz')
    parser.add_argument('--output-dir', type=str,   default=None)
    parser.add_argument('--no-heatmap', action='store_true',
                        help='Skip heatmap generation (faster testing)')
    args = parser.parse_args()

    if args.mode == 'fusion':
        main_fusion(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            npz_path=args.npz,
            output_dir=args.output_dir,
            generate_heatmap=not args.no_heatmap,
        )
    else:
        main()