"""
Comprehensive Evaluation Suite for Multimodal GNN
Generates all metrics needed for paper: ROC AUC, accuracy, precision, recall, F1, confusion matrix
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from multimodal_gnn_fixed import MultimodalToxinClassifier, create_real_dataset
import pandas as pd

class MultimodalEvaluator:
    """Complete evaluation suite for multimodal toxin classifier"""

    def __init__(self, model, class_names=['Calcium', 'Sodium', 'Potassium', 'nonToxin']):
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)

    def predict(self, batch_graphs, esm_embeddings):
        """Get predictions and probabilities from model"""
        self.model.eval()

        with torch.no_grad():
            logits, struct_feat, seq_feat = self.model(batch_graphs, esm_embeddings)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        return predictions.numpy(), probabilities.numpy(), logits.numpy()

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate all classification metrics"""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)

        # ROC AUC (handle case where not all classes are present)
        try:
            if self.num_classes > 2:
                # Multi-class ROC AUC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
                metrics['roc_auc_per_class'] = roc_auc_score(y_true_bin, y_prob, average=None, multi_class='ovr')
            else:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError as e:
            print(f"Warning: Could not calculate ROC AUC - {e}")
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_per_class'] = None

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix_multimodal.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Multimodal GNN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"📊 Confusion matrix saved: {save_path}")
        return cm

    def plot_roc_curves(self, y_true, y_prob, save_path='roc_curves_multimodal.png'):
        """Plot ROC curves for multi-class classification"""
        if self.num_classes < 2:
            print("Warning: Need at least 2 classes for ROC curves")
            return None

        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(self.num_classes):
            if np.sum(y_true_bin[:, i]) > 0:  # Only plot if class exists in data
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multimodal GNN ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"📈 ROC curves saved: {save_path}")

    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("📋 DETAILED CLASSIFICATION REPORT")
        print("="*60)

        report = classification_report(y_true, y_pred,
                                     target_names=self.class_names,
                                     zero_division=0)
        print(report)

    def comprehensive_evaluation(self, batch_graphs, esm_embeddings, y_true, protein_ids):
        """Run complete evaluation suite"""
        print("🔬 Comprehensive Multimodal GNN Evaluation")
        print("="*50)

        # Get predictions
        y_pred, y_prob, logits = self.predict(batch_graphs, esm_embeddings)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)

        # Print summary metrics
        print("\n📊 SUMMARY METRICS")
        print("-"*30)
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision_macro']:.4f} (macro avg)")
        print(f"Recall:       {metrics['recall_macro']:.4f} (macro avg)")
        print(f"F1-Score:     {metrics['f1_macro']:.4f} (macro avg)")

        if metrics['roc_auc_macro'] is not None:
            print(f"ROC AUC:      {metrics['roc_auc_macro']:.4f} (macro avg)")

        # Per-protein predictions
        print(f"\n🧬 PER-PROTEIN PREDICTIONS")
        print("-"*40)
        for i, (pid, true_label, pred_label) in enumerate(zip(protein_ids, y_true, y_pred)):
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = y_prob[i, pred_label]
            status = "✅" if true_label == pred_label else "❌"
            print(f"{status} {pid}: {true_class} → {pred_class} (conf: {confidence:.3f})")

        # Per-class metrics
        print(f"\n📈 PER-CLASS METRICS")
        print("-"*30)
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['precision_per_class']):
                prec = metrics['precision_per_class'][i]
                rec = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                print(f"{class_name:12s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

                if metrics['roc_auc_per_class'] is not None and i < len(metrics['roc_auc_per_class']):
                    auc_score = metrics['roc_auc_per_class'][i]
                    print(f"{'':12s}  AUC={auc_score:.3f}")

        # Generate visualizations
        print(f"\n📊 GENERATING PLOTS...")
        cm = self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curves(y_true, y_prob)

        # Print detailed report
        self.print_classification_report(y_true, y_pred)

        # Save metrics to file
        self.save_metrics_to_file(metrics, protein_ids, y_true, y_pred, y_prob)

        return metrics, y_pred, y_prob

    def save_metrics_to_file(self, metrics, protein_ids, y_true, y_pred, y_prob):
        """Save all metrics to text file"""
        with open('multimodal_evaluation_results.txt', 'w') as f:
            f.write("Multimodal GNN Evaluation Results\n")
            f.write("="*40 + "\n\n")

            # Summary metrics
            f.write("SUMMARY METRICS:\n")
            f.write(f"Accuracy:     {metrics['accuracy']:.4f}\n")
            f.write(f"Precision:    {metrics['precision_macro']:.4f} (macro avg)\n")
            f.write(f"Recall:       {metrics['recall_macro']:.4f} (macro avg)\n")
            f.write(f"F1-Score:     {metrics['f1_macro']:.4f} (macro avg)\n")

            if metrics['roc_auc_macro'] is not None:
                f.write(f"ROC AUC:      {metrics['roc_auc_macro']:.4f} (macro avg)\n")

            f.write(f"\nPER-PROTEIN PREDICTIONS:\n")
            for i, (pid, true_label, pred_label) in enumerate(zip(protein_ids, y_true, y_pred)):
                true_class = self.class_names[true_label]
                pred_class = self.class_names[pred_label]
                confidence = y_prob[i, pred_label]
                status = "✅" if true_label == pred_label else "❌"
                f.write(f"{status} {pid}: {true_class} → {pred_class} (conf: {confidence:.3f})\n")

        print(f"💾 Metrics saved: multimodal_evaluation_results.txt")

def evaluate_trained_model():
    """Load trained model and run comprehensive evaluation"""
    print("🚀 Loading data and running comprehensive evaluation...")

    # Load data
    batch_graphs, esm_embeddings, labels, protein_ids = create_real_dataset()

    # Create and train model (quick training since it's just 3 proteins)
    model = MultimodalToxinClassifier(num_classes=4, hidden_dim=128, fusion_type='concat')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Quick training (10 epochs should be enough for perfect accuracy)
    model.train()
    for epoch in range(10):
        logits, _, _ = model(batch_graphs, esm_embeddings)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Run comprehensive evaluation
    evaluator = MultimodalEvaluator(model)
    metrics, predictions, probabilities = evaluator.comprehensive_evaluation(
        batch_graphs, esm_embeddings, labels.numpy(), protein_ids
    )

    print("\n🎯 Evaluation complete! All plots and metrics saved.")
    print("Files created:")
    print("  📊 confusion_matrix_multimodal.png")
    print("  📈 roc_curves_multimodal.png")
    print("  💾 multimodal_evaluation_results.txt")

    return evaluator, metrics, predictions, probabilities

if __name__ == "__main__":
    evaluator, metrics, predictions, probabilities = evaluate_trained_model()