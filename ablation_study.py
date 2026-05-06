"""
Ablation Study for Multimodal GNN
Tests individual modality contributions:
1. Sequence-only: ESM embeddings → classifier
2. Structure-only: Protein graphs → classifier
3. Multimodal: Both modalities → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multimodal_gnn_fixed import (
    StructuralEncoder, SequenceEncoder, MultimodalToxinClassifier,
    create_real_dataset
)
from multimodal_evaluation import MultimodalEvaluator
import time

class SequenceOnlyClassifier(nn.Module):
    """Classifier using only ESM embeddings (no structural data)"""

    def __init__(self, num_classes=4, hidden_dim=128):
        super(SequenceOnlyClassifier, self).__init__()

        # Reuse sequence encoder from multimodal model
        self.sequence_encoder = SequenceEncoder(hidden_dim=hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, esm_embeddings):
        sequence_features = self.sequence_encoder(esm_embeddings)
        logits = self.classifier(sequence_features)
        return logits, sequence_features, None  # None for struct_feat compatibility

class StructureOnlyClassifier(nn.Module):
    """Classifier using only protein graphs (no sequence data)"""

    def __init__(self, num_classes=4, hidden_dim=128):
        super(StructureOnlyClassifier, self).__init__()

        # Reuse structural encoder from multimodal model
        self.structural_encoder = StructuralEncoder(hidden_dim=hidden_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, batch_graphs):
        structure_features = self.structural_encoder(
            batch_graphs.x,
            batch_graphs.pos,
            batch_graphs.edge_index_spatial,
            batch_graphs.edge_index_disulfide,
            batch_graphs.batch
        )
        logits = self.classifier(structure_features)
        return logits, structure_features, None  # None for seq_feat compatibility

class MultimodalNoDisulfideClassifier(nn.Module):
    """Multimodal classifier WITHOUT special disulfide bond edges"""

    def __init__(self, num_classes=4, hidden_dim=128):
        super(MultimodalNoDisulfideClassifier, self).__init__()

        # Structural encoder that ignores disulfide bonds
        self.structural_encoder = StructuralEncoder(hidden_dim=hidden_dim)
        # Sequence encoder
        self.sequence_encoder = SequenceEncoder(hidden_dim=hidden_dim)

        # Fusion layer (concatenation)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, batch_graphs, esm_embeddings):
        # Structure features WITHOUT using disulfide edges (use empty tensor)
        empty_disulfide = torch.empty((2, 0), dtype=torch.long)

        structure_features = self.structural_encoder(
            batch_graphs.x,
            batch_graphs.pos,
            batch_graphs.edge_index_spatial,  # Only spatial contacts
            empty_disulfide,                  # NO disulfide bonds
            batch_graphs.batch
        )

        # Sequence features
        sequence_features = self.sequence_encoder(esm_embeddings)

        # Fuse both modalities
        combined = torch.cat([structure_features, sequence_features], dim=1)
        fused_features = self.fusion(combined)

        # Classify
        logits = self.classifier(fused_features)
        return logits, structure_features, sequence_features

class AblationTrainer:
    """Trains and evaluates all three model variants"""

    def __init__(self, num_epochs=30, lr=0.001):
        self.num_epochs = num_epochs
        self.lr = lr
        self.class_names = ['Calcium', 'Sodium', 'Potassium', 'nonToxin']

    def train_model(self, model, data_loader_func, model_name):
        """Train a single model variant"""
        print(f"\n🔬 Training {model_name}...")
        print("-" * 40)

        # Load data
        batch_graphs, esm_embeddings, labels, protein_ids = create_real_dataset()

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training tracking
        train_losses = []
        train_accuracies = []

        start_time = time.time()

        for epoch in range(self.num_epochs):
            model.train()

            # Forward pass (different for each model type)
            if model_name == "Sequence-Only":
                logits, _, _ = model(esm_embeddings)
            elif model_name == "Structure-Only":
                logits, _, _ = model(batch_graphs)
            elif model_name == "Multimodal-No-Disulfide":
                logits, _, _ = model(batch_graphs, esm_embeddings)
            elif model_name == "Multimodal-Full":
                logits, _, _ = model(batch_graphs, esm_embeddings)
            else:  # Default multimodal
                logits, _, _ = model(batch_graphs, esm_embeddings)

            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()

            train_losses.append(loss.item())
            train_accuracies.append(accuracy)

            # Show progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch+1:2d}/{self.num_epochs}: Loss={loss.item():.4f}, Acc={accuracy:.3f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")

        return model, train_losses, train_accuracies

    def evaluate_model(self, model, model_name):
        """Evaluate a trained model"""
        print(f"\n📊 Evaluating {model_name}...")

        # Load data
        batch_graphs, esm_embeddings, labels, protein_ids = create_real_dataset()

        model.eval()
        with torch.no_grad():
            # Forward pass
            if model_name == "Sequence-Only":
                logits, features, _ = model(esm_embeddings)
            elif model_name == "Structure-Only":
                logits, features, _ = model(batch_graphs)
            elif model_name == "Multimodal-No-Disulfide":
                logits, features, _ = model(batch_graphs, esm_embeddings)
            elif model_name == "Multimodal-Full":
                logits, features, _ = model(batch_graphs, esm_embeddings)
            else:  # Default multimodal
                logits, features, _ = model(batch_graphs, esm_embeddings)

            # Calculate metrics
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            loss = F.cross_entropy(logits, labels).item()

        # Per-protein results
        print(f"Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {loss:.4f}")

        print(f"  Per-protein predictions:")
        for i, (pid, true_label, pred_label) in enumerate(zip(protein_ids, labels, predictions)):
            true_class = self.class_names[true_label]
            pred_class = self.class_names[pred_label]
            confidence = probabilities[i, pred_label].item()
            status = "✅" if true_label == pred_label else "❌"
            print(f"    {status} {pid}: {true_class} → {pred_class} (conf: {confidence:.3f})")

        return {
            'accuracy': accuracy,
            'loss': loss,
            'predictions': predictions.numpy(),
            'probabilities': probabilities.numpy(),
            'features': features.numpy() if features is not None else None
        }

def run_ablation_study():
    """Run complete ablation study comparing all four approaches"""
    print("🧬 Multimodal GNN Ablation Study")
    print("="*50)
    print("Testing modality contributions:")
    print("1. Sequence-Only (ESM embeddings)")
    print("2. Structure-Only (Protein graphs with disulfide attention)")
    print("3. Multimodal-No-Disulfide (Both modalities, no disulfide attention)")
    print("4. Multimodal-Full (Both modalities with disulfide attention)")

    trainer = AblationTrainer(num_epochs=30, lr=0.001)
    results = {}

    # 1. Sequence-Only Model
    print(f"\n{'='*60}")
    print("1️⃣  SEQUENCE-ONLY MODEL")
    print("="*60)

    sequence_model = SequenceOnlyClassifier(num_classes=4, hidden_dim=128)
    sequence_model, seq_losses, seq_accs = trainer.train_model(
        sequence_model, None, "Sequence-Only"
    )
    seq_results = trainer.evaluate_model(sequence_model, "Sequence-Only")
    results['Sequence-Only'] = seq_results

    # 2. Structure-Only Model
    print(f"\n{'='*60}")
    print("2️⃣  STRUCTURE-ONLY MODEL")
    print("="*60)

    structure_model = StructureOnlyClassifier(num_classes=4, hidden_dim=128)
    structure_model, struct_losses, struct_accs = trainer.train_model(
        structure_model, None, "Structure-Only"
    )
    struct_results = trainer.evaluate_model(structure_model, "Structure-Only")
    results['Structure-Only'] = struct_results

    # 3. Multimodal-No-Disulfide Model
    print(f"\n{'='*60}")
    print("3️⃣  MULTIMODAL-NO-DISULFIDE MODEL")
    print("="*60)

    multimodal_no_disulfide = MultimodalNoDisulfideClassifier(num_classes=4, hidden_dim=128)
    multimodal_no_disulfide, multi_no_losses, multi_no_accs = trainer.train_model(
        multimodal_no_disulfide, None, "Multimodal-No-Disulfide"
    )
    multi_no_results = trainer.evaluate_model(multimodal_no_disulfide, "Multimodal-No-Disulfide")
    results['Multimodal-No-Disulfide'] = multi_no_results

    # 4. Multimodal-Full Model
    print(f"\n{'='*60}")
    print("4️⃣  MULTIMODAL-FULL MODEL")
    print("="*60)

    multimodal_full = MultimodalToxinClassifier(num_classes=4, hidden_dim=128, fusion_type='concat')
    multimodal_full, multi_full_losses, multi_full_accs = trainer.train_model(
        multimodal_full, None, "Multimodal-Full"
    )
    multi_full_results = trainer.evaluate_model(multimodal_full, "Multimodal-Full")
    results['Multimodal-Full'] = multi_full_results

    # 5. Compare Results
    print(f"\n{'='*60}")
    print("📊 ABLATION STUDY RESULTS")
    print("="*60)

    comparison_df = pd.DataFrame({
        'Model': ['Sequence-Only', 'Structure-Only', 'Multimodal-No-Disulfide', 'Multimodal-Full'],
        'Accuracy': [results['Sequence-Only']['accuracy'],
                    results['Structure-Only']['accuracy'],
                    results['Multimodal-No-Disulfide']['accuracy'],
                    results['Multimodal-Full']['accuracy']],
        'Loss': [results['Sequence-Only']['loss'],
                results['Structure-Only']['loss'],
                results['Multimodal-No-Disulfide']['loss'],
                results['Multimodal-Full']['loss']]
    })

    print(comparison_df.round(4))

    # 6. Plot Comparison
    plot_ablation_results(results, [seq_accs, struct_accs, multi_no_accs, multi_full_accs])

    # 7. Save Results
    save_ablation_results(results, comparison_df)

    print(f"\n🎯 Ablation study complete!")
    print("Files created:")
    print("  📊 ablation_comparison.png")
    print("  💾 ablation_results.csv")
    print("  📄 ablation_summary.txt")

    return results, comparison_df

def plot_ablation_results(results, training_curves):
    """Plot ablation study comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy comparison bar plot
    models = ['Sequence-Only', 'Structure-Only', 'Multimodal-No-Disulfide', 'Multimodal-Full']
    accuracies = [results[model]['accuracy'] for model in models]
    colors = ['lightblue', 'lightgreen', 'gold', 'orange']

    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Ablation Study: Model Comparison')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)

    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Training curves
    seq_accs, struct_accs, multi_no_accs, multi_full_accs = training_curves
    epochs = range(1, len(seq_accs) + 1)

    ax2.plot(epochs, seq_accs, 'b-', label='Sequence-Only', linewidth=2)
    ax2.plot(epochs, struct_accs, 'g-', label='Structure-Only', linewidth=2)
    ax2.plot(epochs, multi_no_accs, 'gold', label='Multimodal-No-Disulfide', linewidth=2)
    ax2.plot(epochs, multi_full_accs, 'orange', label='Multimodal-Full', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy')
    ax2.set_title('Training Curves Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_ablation_results(results, comparison_df):
    """Save ablation results to files"""
    # Save CSV
    comparison_df.to_csv('ablation_results.csv', index=False)

    # Save detailed text summary
    with open('ablation_summary.txt', 'w') as f:
        f.write("Multimodal GNN Ablation Study Results\n")
        f.write("="*40 + "\n\n")

        f.write("SUMMARY:\n")
        f.write(comparison_df.round(4).to_string(index=False))
        f.write("\n\n")

        f.write("DETAILED RESULTS:\n\n")
        for model_name, result in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Loss: {result['loss']:.4f}\n\n")

if __name__ == "__main__":
    results, comparison_df = run_ablation_study()