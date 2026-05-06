"""
Statistical Manifold Theory Integration for Multimodal Toxin Classification
Integrates Fisher Information Geometry with existing ESM + GNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('statistical_manifold_agent/')

from information_geometry import (
    FisherInformationMatrix,
    InformationTheoreticFeatureSelection,
    UncertaintyQuantification
)

class ProbabilisticESMEncoder(nn.Module):
    """
    Transform ESM embeddings into probability distributions on ion channel manifold
    """

    def __init__(self, esm_dim=1280, num_classes=4, hidden_dim=256):
        super().__init__()
        self.esm_dim = esm_dim
        self.num_classes = num_classes

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate heads for mean and variance parameters
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)  # Ensure probabilities sum to 1
        )

        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus()  # Ensure positive variances
        )

    def forward(self, esm_embeddings):
        """
        Args:
            esm_embeddings: [batch_size, esm_dim] ESM protein embeddings

        Returns:
            distributions: Normal distribution over ion channel classes
        """
        # Shared feature processing
        features = self.shared_layers(esm_embeddings)

        # Extract distribution parameters
        mu = self.mean_head(features)  # [batch_size, num_classes]
        sigma = self.variance_head(features) + 1e-6  # [batch_size, num_classes]

        # Create multivariate normal distribution
        distribution = Normal(mu, sigma)

        return distribution

class FisherInformationFusion(nn.Module):
    """
    Fuse sequence and structural features using Fisher Information geometry
    """

    def __init__(self, structural_dim=64, fusion_dim=128):
        super().__init__()
        self.structural_dim = structural_dim
        self.fusion_dim = fusion_dim

        # Initialize Fisher Information computer
        self.fisher_computer = FisherInformationMatrix(num_classes=4)

        # Fusion layers (8 distribution params + structural features)
        self.fusion_layers = nn.Sequential(
            nn.Linear(8 + structural_dim, fusion_dim),  # 8 = 4 means + 4 variances
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )

    def extract_distribution_features(self, distributions):
        """
        Extract numerical features from probability distributions

        Args:
            distributions: List of Normal distributions

        Returns:
            features: [batch_size, 8] flattened distribution parameters
        """
        batch_size = len(distributions)
        features = torch.zeros(batch_size, 8)

        for i, dist in enumerate(distributions):
            # Extract mean and variance parameters - ensure single protein
            mu = dist.loc  # Should be [4] for ion channel classes
            sigma = dist.scale  # Should be [4] for ion channel classes

            # If still has batch dimensions, take the i-th element
            if mu.dim() > 1:
                mu = mu[0]  # Take first (and should be only) element
            if sigma.dim() > 1:
                sigma = sigma[0]  # Take first (and should be only) element

            print(f"Debug: dist {i} - mu shape after fix: {mu.shape}, sigma shape: {sigma.shape}")

            features[i] = torch.cat([mu, sigma])  # [8] total features

        return features

    def compute_fisher_distances(self, distributions):
        """
        Compute pairwise Fisher Information distances

        Args:
            distributions: List of Normal distributions

        Returns:
            distance_matrix: [batch_size, batch_size] Fisher distances
        """
        return self.fisher_computer.batch_fisher_distances(distributions)

    def forward(self, sequence_distributions, structural_features):
        """
        Args:
            sequence_distributions: List of Normal distributions from ESM
            structural_features: [batch_size, structural_dim] from GNN

        Returns:
            fused_features: [batch_size, fusion_dim] combined features
            fisher_distances: [batch_size, batch_size] Information distances
        """
        # Extract distribution parameters
        dist_features = self.extract_distribution_features(sequence_distributions)

        # Compute Fisher distances (for analysis)
        fisher_distances = self.compute_fisher_distances(sequence_distributions)

        # Concatenate sequence and structural features
        combined_features = torch.cat([dist_features, structural_features], dim=-1)

        # Fusion processing
        fused_features = self.fusion_layers(combined_features)

        return fused_features, fisher_distances

class StatisticalManifoldToxinClassifier(nn.Module):
    """
    Complete multimodal classifier with Statistical Manifold Theory integration
    """

    def __init__(self, esm_dim=1280, structural_dim=64, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        # Initialize components
        self.probabilistic_encoder = ProbabilisticESMEncoder(esm_dim, num_classes)
        self.fisher_fusion = FisherInformationFusion(structural_dim)
        self.uncertainty_quantifier = UncertaintyQuantification()

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, esm_embeddings, structural_features):
        """
        Args:
            esm_embeddings: [batch_size, esm_dim] ESM protein embeddings
            structural_features: [batch_size, structural_dim] GNN features

        Returns:
            logits: [batch_size, num_classes] class predictions
            uncertainty: [batch_size] prediction uncertainty scores
            distributions: List of probability distributions
            fisher_distances: [batch_size, batch_size] Information distances
        """
        # Transform ESM to probability distributions
        distributions = self.probabilistic_encoder(esm_embeddings)

        # Convert to list for Fisher fusion - properly extract individual distributions
        batch_size = distributions.loc.shape[0]
        dist_list = [Normal(distributions.loc[i], distributions.scale[i])
                    for i in range(batch_size)]

        # Fisher Information fusion
        fused_features, fisher_distances = self.fisher_fusion(dist_list, structural_features)

        # Final classification
        logits = self.classifier(fused_features)

        # Compute uncertainty scores
        uncertainty = torch.tensor([
            self.uncertainty_quantifier.compute_prediction_entropy(dist)
            for dist in dist_list
        ])

        return logits, uncertainty, dist_list, fisher_distances

def integrate_with_existing_pipeline():
    """
    Integration function to connect Statistical Manifold Theory with existing system
    """
    print("🧮 Integrating Statistical Manifold Theory with Multimodal Pipeline 🧮")
    print("=" * 70)

    # Load existing data (using existing format from multimodal_gnn_fixed.py)
    try:
        # Load ESM embeddings
        esm_data = np.load('esm_features_multimodal_dict.npy', allow_pickle=True).item()
        print(f"✅ Loaded ESM embeddings: {len(esm_data)} proteins")

        # Convert to tensor format - handle variable sequence lengths
        protein_ids = list(esm_data.keys())

        # Pool each protein's ESM embedding to fixed size (mean across sequence)
        esm_list = []
        for pid in protein_ids:
            protein_esm = torch.tensor(esm_data[pid], dtype=torch.float32)  # [seq_len, 1280]
            pooled_esm = protein_esm.mean(dim=0)  # [1280] - mean across sequence
            esm_list.append(pooled_esm)

        esm_tensor = torch.stack(esm_list)  # [num_proteins, 1280]
        print(f"   ESM tensor shape after pooling: {esm_tensor.shape}")

    except FileNotFoundError:
        print("❌ ESM embeddings not found. Please run ESM generation first.")
        return None

    # Load real structural features from protein graphs
    try:
        protein_graphs = torch.load('protein_graphs_with_disulfides.pt')
        print(f"✅ Loaded protein graphs: {len(protein_graphs)} proteins")

        # Extract structural features (we need to process the graphs through the GNN)
        # For now, use node feature statistics as structural representation
        structural_features = []
        for pid in protein_ids:
            if pid in protein_graphs:
                graph = protein_graphs[pid]
                # Use statistics of node features as protein-level representation
                node_features = graph.x  # [num_atoms, 8] (element + coordinates)
                structural_feat = torch.cat([
                    node_features.mean(dim=0),  # Mean features
                    node_features.std(dim=0),   # Std features
                    torch.tensor([graph.num_nodes, graph.num_edges], dtype=torch.float32)  # Graph size
                ])
                structural_features.append(structural_feat)
            else:
                print(f"⚠️  No structural data for {pid}")
                return None

        structural_features = torch.stack(structural_features)
        print(f"   Structural features shape: {structural_features.shape}")

    except FileNotFoundError:
        print("❌ Protein graphs not found. Please run disulfide_pyg_builder.py first.")
        return None

    # Initialize Statistical Manifold model
    structural_dim = structural_features.shape[1]  # Use actual structural feature dimension
    model = StatisticalManifoldToxinClassifier(
        esm_dim=1280,
        structural_dim=structural_dim,
        num_classes=4
    )

    print(f"✅ Statistical Manifold model initialized")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits, uncertainty, distributions, fisher_distances = model(
            esm_tensor, structural_features
        )

    print(f"✅ Forward pass successful:")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Uncertainty scores: {uncertainty}")
    print(f"   Fisher distance matrix shape: {fisher_distances.shape}")
    print(f"   Fisher distance matrix:\n{fisher_distances}")

    # Analyze Fisher Information patterns
    print(f"\n🔍 Fisher Information Analysis:")
    for i, (pid, dist) in enumerate(zip(protein_ids, distributions)):
        entropy = model.uncertainty_quantifier.compute_prediction_entropy(dist)
        uncertainty_level = model.uncertainty_quantifier.classify_uncertainty_level(entropy)

        print(f"   {pid}: μ={dist.loc.detach().numpy():.3f}, "
              f"σ={dist.scale.detach().numpy():.3f}, "
              f"uncertainty={uncertainty_level}")

    # Feature importance analysis
    print(f"\n📊 Information-Theoretic Feature Selection:")
    feature_selector = InformationTheoreticFeatureSelection()

    # Real ion channel labels for the 3 test proteins
    # A3DT44: Other, A6YR20: Na, B3SVF0: K (from data alignment)
    real_labels = torch.tensor([3, 1, 2])  # Other, Na, K

    mutual_info = feature_selector.compute_mutual_information(esm_tensor, real_labels)
    top_dims, top_scores = feature_selector.select_informative_dimensions(mutual_info, top_k=10)

    print(f"   Top 5 informative ESM dimensions: {top_dims[:5].tolist()}")
    print(f"   Their information scores: {top_scores[:5].detach().numpy():.4f}")

    analysis = feature_selector.analyze_feature_importance(mutual_info)
    print(f"   Mean importance: {analysis['mean_importance']:.4f}")
    print(f"   Informative fraction: {analysis['informative_fraction']:.2%}")

    print(f"\n🎯 Statistical Manifold integration successful!")
    print("Ready for training with Fisher Information Geometry!")

    return model, esm_tensor, structural_features, fisher_distances

def create_training_loop():
    """
    Training loop specialized for Statistical Manifold Theory
    """
    print("\n🔄 Creating Statistical Manifold Training Loop")

    # Integration components
    result = integrate_with_existing_pipeline()
    if result is None:
        print("❌ Integration failed, cannot proceed with training")
        return None

    model, esm_data, structural_data, _ = result

    # Real ion channel labels for training
    labels = torch.tensor([3, 1, 2])  # A3DT44: Other, A6YR20: Na, B3SVF0: K
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()

        # Forward pass with Fisher Information
        logits, uncertainty, distributions, fisher_distances = model(esm_data, structural_data)

        # Standard classification loss
        loss = criterion(logits, labels)

        # Optional: Add Fisher Information regularization
        # fisher_reg = torch.mean(fisher_distances)  # Encourage informative separations
        # loss += 0.01 * fisher_reg

        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == labels).float().mean()

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}, "
              f"Avg Uncertainty={uncertainty.mean():.4f}")

    print("✅ Statistical Manifold training loop completed!")

    return model

if __name__ == "__main__":
    # Run integration test
    integrate_with_existing_pipeline()

    # Test training
    create_training_loop()