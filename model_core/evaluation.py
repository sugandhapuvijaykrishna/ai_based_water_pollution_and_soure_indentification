"""
Model evaluation: confusion matrix, classification report, visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, class_names: list = None):
        self.class_names = class_names or ['Clean Water', 'Moderate Pollution', 'High Pollution']
    
    def evaluate(self, y_true: np.ndarray, 
                y_pred_probs: np.ndarray) -> dict:
        """
        Compute comprehensive metrics
        
        Args:
            y_true: True labels (N,)
            y_pred_probs: Prediction probabilities (N, num_classes)
        
        Returns:
            Dictionary with metrics
        """
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_true': y_true,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return metrics
    
    def print_report(self, y_true: np.ndarray, 
                    y_pred_probs: np.ndarray):
        """Print detailed classification report"""
        metrics = self.evaluate(y_true, y_pred_probs)
        y_pred = metrics['y_pred']
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        print("\nPer-Class Metrics:")
        print(classification_report(y_true, y_pred, 
                                   target_names=self.class_names,
                                   digits=4))
        
        print("="*70 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, metrics: dict, 
                             figsize: tuple = (8, 6)):
        """Plot confusion matrix heatmap"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to confusion_matrix.png")
        plt.show()
    
    def plot_training_history(self, history: dict, 
                             figsize: tuple = (14, 5)):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1-Score (if available)
        if 'recall' in history:
            axes[2].plot(history['recall'], label='Train Recall', linewidth=2)
            axes[2].plot(history['val_recall'], label='Val Recall', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Recall')
            axes[2].set_title('Recall Curves')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history saved to training_history.png")
        plt.show()
    
    @staticmethod
    def plot_prediction_uncertainty(y_pred_probs: np.ndarray, 
                                   y_true: np.ndarray,
                                   figsize: tuple = (10, 5)):
        """Plot prediction confidence"""
        max_probs = np.max(y_pred_probs, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        correct = y_pred == y_true
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Confidence histogram
        axes[0].hist(max_probs[correct], bins=30, alpha=0.7, label='Correct', color='green')
        axes[0].hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0].set_xlabel('Prediction Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Prediction Confidence Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ROC-like curve
        sorted_idx = np.argsort(max_probs)[::-1]
        accuracy_at_conf = []
        confidence_levels = np.linspace(0, 1, 20)
        
        for conf_threshold in confidence_levels:
            mask = max_probs >= conf_threshold
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                accuracy_at_conf.append(acc)
            else:
                accuracy_at_conf.append(np.nan)
        
        axes[1].plot(confidence_levels, accuracy_at_conf, marker='o', linewidth=2)
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy vs Confidence Threshold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig('prediction_confidence.png', dpi=300, bbox_inches='tight')
        print("✓ Confidence plot saved to prediction_confidence.png")
        plt.show()

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Mock predictions
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.random.rand(6, 3)  # softmax outputs
    
    metrics = evaluator.print_report(y_true, y_pred)
    evaluator.plot_confusion_matrix(metrics)