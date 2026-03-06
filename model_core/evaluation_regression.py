"""
Regression-specific evaluation and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class RegressionVisualization:
    """Advanced visualization for regression results"""
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_training_history(self, 
                             history: dict,
                             figsize: tuple = (16, 5)):
        """
        Plot training and validation curves
        
        Args:
            history: Training history dictionary
            figsize: Figure size
        """
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Loss
        axes[0].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # MAE
        axes[1].plot(epochs, history['mae'], 'b-', label='Train MAE', linewidth=2)
        axes[1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('MAE', fontsize=11)
        axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # RMSE
        axes[2].plot(epochs, history['rmse'], 'b-', label='Train RMSE', linewidth=2)
        axes[2].plot(epochs, history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('RMSE', fontsize=11)
        axes[2].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # MSE
        axes[3].plot(epochs, history['mse'], 'b-', label='Train MSE', linewidth=2)
        axes[3].plot(epochs, history['val_mse'], 'r-', label='Val MSE', linewidth=2)
        axes[3].set_xlabel('Epoch', fontsize=11)
        axes[3].set_ylabel('MSE', fontsize=11)
        axes[3].set_title('Mean Squared Error', fontsize=12, fontweight='bold')
        axes[3].legend(fontsize=10)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history_regression.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Training history saved")
        plt.show()
    
    def plot_predictions_distribution(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     figsize: tuple = (14, 5)):
        """
        Plot distribution of true vs predicted values
        
        Args:
            y_true: True values (0-100 scale)
            y_pred: Predicted values (0-100 scale)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # True distribution
        axes[0].hist(y_true, bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[0].set_xlabel('Pollution Index', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('True Values Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axvline(np.mean(y_true), color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {np.mean(y_true):.2f}")
        axes[0].legend()
        
        # Predicted distribution
        axes[1].hist(y_pred, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Pollution Index', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Predicted Values Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axvline(np.mean(y_pred), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(y_pred):.2f}")
        axes[1].legend()
        
        # Overlay
        axes[2].hist(y_true, bins=30, alpha=0.6, label='True', color='blue', edgecolor='black')
        axes[2].hist(y_pred, bins=30, alpha=0.6, label='Predicted', color='green', edgecolor='black')
        axes[2].set_xlabel('Pollution Index', fontsize=11)
        axes[2].set_ylabel('Frequency', fontsize=11)
        axes[2].set_title('Overlay Distribution', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions_distribution.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Distribution plot saved")
        plt.show()
    
    def plot_error_analysis(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           figsize: tuple = (15, 10)):
        """
        Comprehensive error analysis
        
        Args:
            y_true: True values (0-100 scale)
            y_pred: Predicted values (0-100 scale)
            figsize: Figure size
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        pct_errors = (abs_errors / (y_true + 1e-6)) * 100
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Scatter: True vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=40, color='blue')
        axes[0, 0].plot([0, 100], [0, 100], 'r--', lw=2.5, label='Perfect Fit')
        axes[0, 0].set_xlabel('True Pollution Index', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Pollution Index', fontsize=11)
        axes[0, 0].set_title('Predictions vs Ground Truth', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlim([-5, 105])
        axes[0, 0].set_ylim([-5, 105])
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        axes[0, 1].scatter(y_pred, errors, alpha=0.6, s=40, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2.5)
        axes[0, 1].fill_between(y_pred, -10, 10, alpha=0.1, color='gray',
                               label='±10 range')
        axes[0, 1].set_xlabel('Predicted Pollution Index', fontsize=11)
        axes[0, 1].set_ylabel('Residuals (True - Pred)', fontsize=11)
        axes[0, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([-50, 50])
        
        # Error by predicted value
        axes[0, 2].scatter(y_pred, abs_errors, alpha=0.6, s=40, color='orange')
        axes[0, 2].set_xlabel('Predicted Pollution Index', fontsize=11)
        axes[0, 2].set_ylabel('Absolute Error', fontsize=11)
        axes[0, 2].set_title('Error Magnitude by Prediction', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Error distribution histogram
        axes[1, 0].hist(errors, bins=40, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].axvline(0, color='r', linestyle='--', lw=2.5, label='Zero Error')
        axes[1, 0].axvline(np.mean(errors), color='g', linestyle='--', lw=2.5,
                          label=f'Mean: {np.mean(errors):.2f}')
        axes[1, 0].set_xlabel('Error Value', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Absolute error histogram
        axes[1, 1].hist(abs_errors, bins=40, edgecolor='black', alpha=0.7, color='red')
        axes[1, 1].axvline(np.mean(abs_errors), color='g', linestyle='--', lw=2.5,
                          label=f'MAE: {np.mean(abs_errors):.2f}')
        axes[1, 1].set_xlabel('Absolute Error', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Percentage error
        valid_mask = y_true > 5
        if np.sum(valid_mask) > 0:
            axes[1, 2].hist(pct_errors[valid_mask], bins=40, edgecolor='black', 
                           alpha=0.7, color='brown')
            axes[1, 2].axvline(np.mean(pct_errors[valid_mask]), color='g', 
                              linestyle='--', lw=2.5,
                              label=f'Mean: {np.mean(pct_errors[valid_mask]):.2f}%')
            axes[1, 2].set_xlabel('Percentage Error (%)', fontsize=11)
            axes[1, 2].set_ylabel('Frequency', fontsize=11)
            axes[1, 2].set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
            axes[1, 2].legend(fontsize=10)
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Error analysis saved")
        plt.show()
    
    def plot_quantile_regression(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                figsize: tuple = (14, 5)):
        """
        Quantile-based analysis and calibration curves
        
        Args:
            y_true: True values
            y_pred: Predicted values
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Q-Q plot (sorted values)
        sorted_true = np.sort(y_true)
        sorted_pred = np.sort(y_pred)
        
        axes[0].scatter(sorted_true, sorted_pred, alpha=0.6, s=40, color='blue')
        axes[0].plot([0, 100], [0, 100], 'r--', lw=2.5, label='Perfect Fit')
        axes[0].set_xlabel('True (Sorted)', fontsize=11)
        axes[0].set_ylabel('Predicted (Sorted)', fontsize=11)
        axes[0].set_title('Quantile-Quantile Plot', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Calibration plot: mean prediction vs mean true by bins
        n_bins = 10
        indices = np.argsort(y_pred)
        bin_size = len(y_pred) // n_bins
        
        bin_pred_means = []
        bin_true_means = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_pred)
            
            bin_indices = indices[start_idx:end_idx]
            bin_pred_means.append(np.mean(y_pred[bin_indices]))
            bin_true_means.append(np.mean(y_true[bin_indices]))
        
        axes[1].scatter(bin_pred_means, bin_true_means, s=100, alpha=0.7, color='green')
        axes[1].plot([0, 100], [0, 100], 'r--', lw=2.5, label='Perfect Calibration')
        axes[1].set_xlabel('Mean Predicted Pollution Index', fontsize=11)
        axes[1].set_ylabel('Mean True Pollution Index', fontsize=11)
        axes[1].set_title('Calibration Plot (Binned)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([-5, 105])
        axes[1].set_ylim([-5, 105])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quantile_calibration.png',
                   dpi=300, bbox_inches='tight')
        print("✓ Quantile plot saved")
        plt.show()


if __name__ == "__main__":
    # Example usage
    import json
    
    # Create dummy history
    epochs = 50
    history = {
        'loss': np.linspace(0.1, 0.01, epochs),
        'val_loss': np.linspace(0.12, 0.015, epochs),
        'mae': np.linspace(0.3, 0.05, epochs),
        'val_mae': np.linspace(0.35, 0.06, epochs),
        'rmse': np.linspace(0.4, 0.08, epochs),
        'val_rmse': np.linspace(0.45, 0.09, epochs),
        'mse': np.linspace(0.1, 0.01, epochs),
        'val_mse': np.linspace(0.12, 0.012, epochs)
    }
    
    # Create dummy predictions
    y_true = np.random.uniform(10, 80, 500)
    y_pred = y_true + np.random.normal(0, 5, 500)
    y_pred = np.clip(y_pred, 0, 100)
    
    # Visualize
    viz = RegressionVisualization()
    viz.plot_training_history(history)
    viz.plot_predictions_distribution(y_true, y_pred)
    viz.plot_error_analysis(y_true, y_pred)
    viz.plot_quantile_regression(y_true, y_pred)