#!/usr/bin/env python3
"""
Full Dataset Training Script for Multimodal GNN
Trains on all 1,376 proteins with perfect FASTA-based ESM features and structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Batch
import pickle
import time
from pathlib import Path

# Import existing multimodal architecture
from multimodal_gnn_fixed import MultimodalToxinClassifier

def load_fasta_based_data():
    """Load the perfectly aligned FASTA-based dataset"""
    print("=== Loading FASTA-Based Dataset ===")

    # Load ESM features
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    print(f"Loaded ESM features: {len(esm_features)} proteins")

    # Load metadata
    metadata_df = pd.read_csv('feature_metadata_fasta_aligned.csv')
    print(f"Loaded metadata: {len(metadata_df)} entries")

    # Verify alignment
    assert len(esm_features) == len(metadata_df), "ESM features and metadata must match!"

    return esm_features, metadata_df

def determine_ion_channel_class(row):
    """cone label logic from marc.py - handle Zhang vs match conflicts"""
    if row['match'] == row['Zhang result']:
        return row['match']
    elif row['match'] == 'Other':
        return row['Zhang result']  # Prefer specific classification
    elif row['Zhang result'] == 'Other':
        return row['match']  # Prefer specific classification
    else:
        print(f"CONFLICT: {row['Entry']} - match='{row['match']}' vs Zhang='{row['Zhang result']}' - EXCLUDED")
        return None  # Conflicting specific classifications

LABEL_NORMALIZE = {
    'Ca': 'Calcium',
    'Na': 'Sodium',
    'K':  'Potassium',
}

def create_labels_from_metadata(metadata_df):
    """Extract and encode classification labels with proper cone handling"""
    print("=== Creating Labels with cone Logic ===")

    # Load datasets to get target classes
    cone_df = pd.read_excel('cone_trainingData.xlsx')
    toxin_df = pd.read_excel('toxinCombined_Finalv3.xlsx')

    labels = []
    valid_proteins = []

    for _, row in metadata_df.iterrows():
        protein_id = row['protein_id']
        dataset = row['dataset']

        if dataset == 'cone':
            # Find in cone dataset and apply marc.py logic
            protein_row = cone_df[cone_df['Entry'] == protein_id]
            if len(protein_row) > 0:
                target_class = determine_ion_channel_class(protein_row.iloc[0])
                if target_class is not None:  # Skip conflicting classifications
                    target_class = LABEL_NORMALIZE.get(target_class, target_class)
                    labels.append(target_class)
                    valid_proteins.append(row)

        elif dataset == 'toxin_combined':
            # Find in ToxinCombined dataset
            protein_row = toxin_df[toxin_df['entry'] == protein_id]
            if len(protein_row) > 0:
                target_class = protein_row.iloc[0]['ionChannelClass']
                labels.append(target_class)
                valid_proteins.append(row)

    print(f"Found labels for {len(labels)} proteins (excluded conflicting cone entries)")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Print class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    print(f"\nEncoded classes: {len(label_encoder.classes_)} classes")
    print(f"Classes: {label_encoder.classes_}")

    return encoded_labels, label_encoder, valid_proteins

def parse_pdb_to_tensors(pdb_path: str):
    """Parse PDB file to PyTorch tensors (from disulfide_pyg_builder.py)"""
    atoms_data = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip() or atom_name[0]

                atoms_data.append({
                    'atom_id': atom_id,
                    'atom_name': atom_name,
                    'residue_name': residue_name,
                    'residue_num': residue_num,
                    'x': x, 'y': y, 'z': z,
                    'element': element
                })

    # Convert to tensors
    coordinates = torch.tensor([[atom['x'], atom['y'], atom['z']] for atom in atoms_data], dtype=torch.float)

    # Simple node features: element type encoding + coordinates
    unique_elements = ['C', 'N', 'O', 'S', 'H']  # Common protein elements
    element_to_idx = {elem: i for i, elem in enumerate(unique_elements)}

    node_features = []
    for atom in atoms_data:
        element = atom['element']
        # One-hot element encoding + 3D coordinates
        element_idx = element_to_idx.get(element, 0)  # Default to C if unknown
        one_hot = [0] * len(unique_elements)
        one_hot[element_idx] = 1

        # Combine element encoding with coordinates
        features = one_hot + [atom['x'], atom['y'], atom['z']]
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float)
    return node_features, coordinates, atoms_data

def detect_disulfide_bonds(coordinates: torch.Tensor, atoms_data: list, threshold: float = 2.5):
    """Detect disulfide bonds between cysteine sulfur atoms"""
    cys_sulfur_indices = []

    # Find all cysteine sulfur atoms
    for i, atom in enumerate(atoms_data):
        if atom['residue_name'] == 'CYS' and atom['atom_name'] == 'SG':
            cys_sulfur_indices.append(i)

    disulfide_bonds = []
    for i in range(len(cys_sulfur_indices)):
        for j in range(i + 1, len(cys_sulfur_indices)):
            idx1, idx2 = cys_sulfur_indices[i], cys_sulfur_indices[j]

            coord1 = coordinates[idx1]
            coord2 = coordinates[idx2]
            distance = torch.norm(coord1 - coord2).item()

            if distance <= threshold:
                disulfide_bonds.append((idx1, idx2))

    return disulfide_bonds

def build_spatial_edges(coordinates: torch.Tensor, disulfide_pairs: list, threshold: float = 4.0):
    """Build spatial contact edges (excluding disulfide bonds)"""
    num_atoms = coordinates.shape[0]
    spatial_edges = []

    # Convert disulfide pairs to set for fast lookup
    disulfide_set = set(disulfide_pairs) | set((j, i) for i, j in disulfide_pairs)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if (i, j) in disulfide_set:
                continue

            distance = torch.norm(coordinates[i] - coordinates[j]).item()
            if distance <= threshold:
                spatial_edges.extend([(i, j), (j, i)])  # Undirected

    if len(spatial_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(spatial_edges, dtype=torch.long).t()

def build_disulfide_edges(disulfide_bonds: list):
    """Build disulfide bond edges"""
    if len(disulfide_bonds) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    disulfide_edges = []
    for i, j in disulfide_bonds:
        disulfide_edges.extend([(i, j), (j, i)])  # Undirected

    return torch.tensor(disulfide_edges, dtype=torch.long).t()

def create_protein_graph(protein_info):
    """Create a single protein graph from structure file"""
    pdb_path = protein_info['structure_file']

    try:
        # Parse PDB file
        node_features, coordinates, atoms_data = parse_pdb_to_tensors(pdb_path)

        # Detect disulfide bonds
        disulfide_bonds = detect_disulfide_bonds(coordinates, atoms_data)

        # Build edges
        spatial_edges = build_spatial_edges(coordinates, disulfide_bonds)
        disulfide_edges = build_disulfide_edges(disulfide_bonds)

        # node_features is [num_atoms, 8] (5 element one-hot + 3 xyz already bundled)
        # Model expects x=[num_atoms, 5] and pos=[num_atoms, 3] separately,
        # then concatenates them itself — so split here to avoid double-counting coords
        graph = Data(
            x=node_features[:, :5],
            pos=coordinates,
            edge_index_spatial=spatial_edges,
            edge_index_disulfide=disulfide_edges,
        )
        return graph

    except Exception as e:
        print(f"Error processing {protein_info['protein_id']}: {e}")
        return None

def create_protein_graphs_batch(valid_proteins):
    """Create protein graphs for all valid proteins"""
    print("=== Creating Protein Graphs ===")

    graphs = []
    failed_count = 0

    for i, protein_info in enumerate(valid_proteins):
        if (i + 1) % 100 == 0:
            print(f"Processing {i+1}/{len(valid_proteins)} proteins...")

        graph = create_protein_graph(protein_info)
        if graph is not None:
            graphs.append(graph)
        else:
            failed_count += 1

    print(f"Successfully created {len(graphs)} graphs ({failed_count} failed)")
    return graphs

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (graph_batch, esm_batch, labels_batch) in enumerate(train_loader):
        graph_batch = graph_batch.to(device)
        esm_batch = esm_batch.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        # Forward pass (MULTIMODAL: structure + sequence)
        logits, _, _ = model(graph_batch, esm_batch)
        loss = criterion(logits, labels_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels_batch).sum().item()
        total_samples += labels_batch.size(0)
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device, label_encoder):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for graph_batch, esm_batch, labels_batch in test_loader:
            graph_batch = graph_batch.to(device)
            esm_batch = esm_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Forward pass (MULTIMODAL: structure + sequence)
            logits, _, _ = model(graph_batch, esm_batch)
            loss = criterion(logits, labels_batch)

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=label_encoder.classes_
    ))

    return avg_loss, accuracy, all_predictions, all_labels

def main():
    """Main training function"""
    print("=== Full Dataset Multimodal GNN Training ===")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    esm_features, metadata_df = load_fasta_based_data()
    labels, label_encoder, valid_proteins = create_labels_from_metadata(metadata_df)

    print(f"Dataset size: {len(valid_proteins)} proteins with labels")

    # Create protein graphs
    protein_graphs = create_protein_graphs_batch(valid_proteins)

    # Filter out failed graphs and corresponding data
    successful_indices = [i for i, graph in enumerate(protein_graphs) if graph is not None]
    protein_graphs = [protein_graphs[i] for i in successful_indices]
    valid_proteins = [valid_proteins[i] for i in successful_indices]
    labels = [labels[i] for i in successful_indices]

    print(f"Successfully processed: {len(protein_graphs)} proteins")

    # Convert ESM features to tensors for successful proteins
    esm_tensors = []
    for protein_info in valid_proteins:
        feature_key = protein_info['feature_key']
        esm_tensor = torch.from_numpy(esm_features[feature_key]).float()
        esm_tensors.append(esm_tensor)

    esm_tensors = torch.stack(esm_tensors)
    labels_tensor = torch.from_numpy(np.array(labels)).long()

    print(f"ESM features shape: {esm_tensors.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Protein graphs: {len(protein_graphs)}")

    # Train/test split
    train_indices, test_indices = train_test_split(
        range(len(protein_graphs)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Prepare data for loaders
    train_graphs = [protein_graphs[i] for i in train_indices]
    test_graphs = [protein_graphs[i] for i in test_indices]

    train_esm = esm_tensors[train_indices]
    test_esm = esm_tensors[test_indices]

    train_labels = labels_tensor[train_indices]
    test_labels = labels_tensor[test_indices]

    # Custom dataset class for graphs + ESM features
    class MultimodalDataset(torch.utils.data.Dataset):
        def __init__(self, graphs, esm_features, labels):
            self.graphs = graphs
            self.esm_features = esm_features
            self.labels = labels

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx], self.esm_features[idx], self.labels[idx]

    def collate_fn(batch):
        """Custom collate function for batching graphs and ESM features"""
        graphs, esm_features, labels = zip(*batch)

        # Batch graphs using PyG
        batched_graphs = Batch.from_data_list(graphs)

        # Stack ESM features and labels
        esm_batch = torch.stack(esm_features)
        labels_batch = torch.stack(labels)

        return batched_graphs, esm_batch, labels_batch

    # Create datasets
    train_dataset = MultimodalDataset(train_graphs, train_esm, train_labels)
    test_dataset = MultimodalDataset(test_graphs, test_esm, test_labels)

    # Create data loaders with custom collate function
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = MultimodalToxinClassifier(
        esm_dim=1280,
        gnn_hidden_dim=256,
        num_classes=num_classes,
        fusion_hidden_dim=512,
        dropout_rate=0.1
    ).to(device)

    print(f"Model initialized for {num_classes} classes")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_counts  = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Training loop
    num_epochs       = 50
    patience         = 10
    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0
    train_losses, train_accuracies, val_losses, val_accs = [], [], [], []

    print(f"\n=== Starting Training (up to {num_epochs} epochs, patience={patience}) ===")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device, label_encoder)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  train {train_loss:.4f}/{train_acc*100:.1f}%  val {val_loss:.4f}/{val_acc*100:.1f}%  ({epoch_time:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss; best_epoch = epoch + 1; patience_counter = 0
            torch.save(model.state_dict(), 'multimodal_gnn_full_dataset.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")

    model.load_state_dict(torch.load('multimodal_gnn_full_dataset.pth'))
    print(f"Loaded best model from epoch {best_epoch}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, device, label_encoder
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save results
    print("\n=== Saving Results ===")

    # Save model
    torch.save(model.state_dict(), 'multimodal_gnn_full_dataset.pth')
    print("Model saved: multimodal_gnn_full_dataset.pth")

    # Save label encoder
    with open('label_encoder_full_dataset.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved: label_encoder_full_dataset.pkl")

    # Save training curves data (plotting disabled due to library issues)
    training_data = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
    }
    with open('training_curves_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    print("Training curves data saved: training_curves_data.pkl")

    print("\n=== Training Complete! ===")
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()