#!/usr/bin/env python3
"""
Spectral Multimodal GNN — 3-modality training (Structure + Sequence + Spectral)
Mirrors train_full_multimodal_v2.py but adds spectral graph features as a 3rd modality.
Spectral features: 15D Laplacian eigenvalue spectrum per protein graph.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spectral_agent'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pickle
import time
from pathlib import Path

from spectral_features import extract_spectral_features
from multimodal_gnn_fixed import StructuralEncoder, SequenceEncoder

# ── Model ─────────────────────────────────────────────────────────────────────

class SpectralEncoder(nn.Module):
    def __init__(self, spectral_dim=16, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, spectral_features):
        return self.projection(spectral_features)


class SpectralMultimodalClassifier(nn.Module):
    def __init__(self, num_classes=4, esm_dim=1280, gnn_hidden_dim=256,
                 spectral_dim=16, fusion_hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.structural_encoder = StructuralEncoder(hidden_dim=gnn_hidden_dim)
        self.sequence_encoder   = SequenceEncoder(esm_dim=esm_dim, hidden_dim=gnn_hidden_dim)
        self.spectral_encoder   = SpectralEncoder(spectral_dim=spectral_dim, hidden_dim=gnn_hidden_dim)

        # Concat 3 modalities: 3 * gnn_hidden_dim → fusion_hidden_dim → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 3, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, batch_graphs, esm_embeddings, spectral_feats):
        structure_out = self.structural_encoder(
            batch_graphs.x,
            batch_graphs.pos,
            batch_graphs.edge_index_spatial,
            batch_graphs.edge_index_disulfide,
            batch_graphs.batch
        )
        sequence_out = self.sequence_encoder(esm_embeddings)
        spectral_out = self.spectral_encoder(spectral_feats)

        combined = torch.cat([structure_out, sequence_out, spectral_out], dim=1)
        logits = self.classifier(combined)
        return logits, structure_out, sequence_out, spectral_out

# ── Data loading (identical to v2) ────────────────────────────────────────────

LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}

def load_fasta_based_data():
    print("=== Loading FASTA-Based Dataset ===")
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
    print(f"Loaded ESM features: {len(esm_features)} proteins")
    assert len(esm_features) == len(metadata_df)
    return esm_features, metadata_df


def determine_ion_channel_class(row):
    if row['match'] == row['Zhang result']:
        return row['match']
    elif row['match'] == 'Other':
        return row['Zhang result']
    elif row['Zhang result'] == 'Other':
        return row['match']
    else:
        print(f"CONFLICT: {row['Entry']} — excluded")
        return None


def create_labels_from_metadata(metadata_df):
    print("=== Creating Labels ===")
    cone_df = pd.read_excel('cone_trainingData.xlsx')
    toxin_df = pd.read_excel('toxinCombined_Finalv3.xlsx')

    labels, valid_proteins = [], []
    for _, row in metadata_df.iterrows():
        pid, dataset = row['protein_id'], row['dataset']
        if dataset == 'cone':
            match = cone_df[cone_df['Entry'] == pid]
            if len(match) > 0:
                cls = determine_ion_channel_class(match.iloc[0])
                if cls is not None:
                    labels.append(LABEL_NORMALIZE.get(cls, cls))
                    valid_proteins.append(row)
        elif dataset == 'toxin_combined':
            match = toxin_df[toxin_df['entry'] == pid]
            if len(match) > 0:
                labels.append(match.iloc[0]['ionChannelClass'])
                valid_proteins.append(row)

    print(f"Labels found: {len(labels)} proteins")
    label_encoder   = LabelEncoder()
    encoded_labels  = label_encoder.fit_transform(labels)
    unique, counts  = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")
    print(f"Classes: {label_encoder.classes_}")
    return encoded_labels, label_encoder, valid_proteins

# ── Graph construction (identical to v2) ─────────────────────────────────────

def parse_pdb_to_tensors(pdb_path):
    atoms_data = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms_data.append({
                    'atom_name':    line[12:16].strip(),
                    'residue_name': line[17:20].strip(),
                    'residue_num':  int(line[22:26].strip()),
                    'x': float(line[30:38]), 'y': float(line[38:46]), 'z': float(line[46:54]),
                    'element':      line[76:78].strip() or line[12:16].strip()[0]
                })
    coords = torch.tensor([[a['x'], a['y'], a['z']] for a in atoms_data], dtype=torch.float)
    elem_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'H': 4}
    node_feats = []
    for a in atoms_data:
        oh = [0] * 5
        oh[elem_map.get(a['element'], 0)] = 1
        node_feats.append(oh)
    return torch.tensor(node_feats, dtype=torch.float), coords, atoms_data


def detect_disulfide_bonds(coords, atoms_data, threshold=2.5):
    cys_sg = [i for i, a in enumerate(atoms_data)
              if a['residue_name'] == 'CYS' and a['atom_name'] == 'SG']
    bonds = []
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            if torch.norm(coords[cys_sg[i]] - coords[cys_sg[j]]).item() <= threshold:
                bonds.append((cys_sg[i], cys_sg[j]))
    return bonds


def build_spatial_edges(coords, disulfide_pairs, threshold=4.0):
    n = coords.shape[0]
    dsulf_set = set(disulfide_pairs) | {(j, i) for i, j in disulfide_pairs}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in dsulf_set and torch.norm(coords[i] - coords[j]).item() <= threshold:
                edges.extend([(i, j), (j, i)])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t()


def build_disulfide_edges(bonds):
    if not bonds:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [(i, j) for i, j in bonds] + [(j, i) for i, j in bonds]
    return torch.tensor(edges, dtype=torch.long).t()


def create_protein_graph(protein_info):
    try:
        node_feats, coords, atoms = parse_pdb_to_tensors(protein_info['structure_file'])
        dsulf = detect_disulfide_bonds(coords, atoms)
        return Data(
            x=node_feats,
            pos=coords,
            edge_index_spatial=build_spatial_edges(coords, dsulf),
            edge_index_disulfide=build_disulfide_edges(dsulf),
        )
    except Exception as e:
        print(f"Error {protein_info['protein_id']}: {e}")
        return None

# ── Spectral feature computation ──────────────────────────────────────────────

SPECTRAL_CACHE = 'spectral_features_cache.pt'

def compute_spectral_features_batch(graphs, valid_proteins):
    if Path(SPECTRAL_CACHE).exists():
        print(f"=== Loading Spectral Features from cache ({SPECTRAL_CACHE}) ===")
        spectral_tensor = torch.load(SPECTRAL_CACHE)
        print(f"Loaded: {spectral_tensor.shape}")
        return spectral_tensor

    print("=== Computing Spectral Features (first run — will cache to disk) ===")
    spectral_list = []
    t0 = time.time()
    for i, (graph, pinfo) in enumerate(zip(graphs, valid_proteins)):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(graphs)} proteins  ({elapsed:.0f}s elapsed)")
        feats = extract_spectral_features(
            graph.edge_index_spatial,
            graph.num_nodes,
            protein_id=pinfo['protein_id']
        )
        spectral_list.append(feats)
    spectral_tensor = torch.tensor(np.stack(spectral_list), dtype=torch.float)
    torch.save(spectral_tensor, SPECTRAL_CACHE)
    print(f"Spectral features: {spectral_tensor.shape}  ({time.time()-t0:.1f}s) — cached to {SPECTRAL_CACHE}")
    return spectral_tensor

# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (g, esm, spec, y) in enumerate(loader):
        g, esm, spec, y = g.to(device), esm.to(device), spec.to(device), y.to(device)
        optimizer.zero_grad()
        logits, *_ = model(g, esm, spec)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, label_encoder, verbose=False):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for g, esm, spec, y in loader:
            g, esm, spec, y = g.to(device), esm.to(device), spec.to(device), y.to(device)
            logits, *_ = model(g, esm, spec)
            total_loss += criterion(logits, y).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(trues)).mean()
    if verbose:
        print(classification_report(trues, preds, target_names=label_encoder.classes_))
    return total_loss / len(loader), acc, preds, trues

# ── Dataset / collate ─────────────────────────────────────────────────────────

class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, esm, spectral, labels):
        self.graphs   = graphs
        self.esm      = esm
        self.spectral = spectral
        self.labels   = labels

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.esm[idx], self.spectral[idx], self.labels[idx]


def collate_fn(batch):
    graphs, esm, spectral, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(esm),
        torch.stack(spectral),
        torch.stack(labels)
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Spectral Multimodal GNN Training (Structure + Sequence + Spectral) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    esm_features, metadata_df = load_fasta_based_data()
    labels, label_encoder, valid_proteins = create_labels_from_metadata(metadata_df)
    print(f"Dataset: {len(valid_proteins)} proteins")

    # Build graphs
    print("=== Building Protein Graphs ===")
    graphs, filtered_proteins, filtered_labels = [], [], []
    for i, pinfo in enumerate(valid_proteins):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(valid_proteins)}...")
        g = create_protein_graph(pinfo)
        if g is not None:
            graphs.append(g)
            filtered_proteins.append(pinfo)
            filtered_labels.append(labels[i])
    print(f"Graphs built: {len(graphs)}")

    # ESM tensors for successful proteins
    esm_tensors = torch.stack([
        torch.from_numpy(esm_features[p['feature_key']]).float()
        for p in filtered_proteins
    ])

    # Spectral features
    spectral_tensors = compute_spectral_features_batch(graphs, filtered_proteins)

    labels_tensor = torch.from_numpy(np.array(filtered_labels)).long()
    print(f"ESM: {esm_tensors.shape}, Spectral: {spectral_tensors.shape}, Labels: {labels_tensor.shape}")

    # Train / test split
    train_idx, test_idx = train_test_split(
        range(len(graphs)), test_size=0.2, random_state=42, stratify=filtered_labels
    )

    train_ds = SpectralDataset(
        [graphs[i] for i in train_idx], esm_tensors[train_idx],
        spectral_tensors[train_idx], labels_tensor[train_idx]
    )
    test_ds = SpectralDataset(
        [graphs[i] for i in test_idx], esm_tensors[test_idx],
        spectral_tensors[test_idx], labels_tensor[test_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    num_classes = len(label_encoder.classes_)
    model = SpectralMultimodalClassifier(
        num_classes=num_classes,
        esm_dim=1280,
        gnn_hidden_dim=256,
        spectral_dim=16,
        fusion_hidden_dim=512,
        dropout_rate=0.1
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class-weighted loss (same protocol as all other models)
    train_labels_np = labels_tensor[list(train_idx)].numpy()
    class_counts = np.bincount(train_labels_np)
    class_weights = 1.0 / class_counts.astype(np.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training loop — 50 epochs, early stopping patience=10
    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    print(f"\n=== Training for up to {num_epochs} epochs (early stop patience={patience}) ===")
    t0 = time.time()

    for epoch in range(num_epochs):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, device, label_encoder)
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(vl_loss);   val_accs.append(vl_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}  "
                  f"({time.time()-t_ep:.1f}s)")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch + 1
            import copy
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= patience:
            print(f"Early stop at epoch {epoch+1} (best epoch {best_epoch})")
            break

    print(f"\nBest epoch: {best_epoch}  best_val_loss={best_val_loss:.4f}")
    model.load_state_dict(best_state)
    print(f"Total training time: {time.time()-t0:.1f}s")

    # Final report
    print("\n=== Final Evaluation ===")
    test_loss, test_acc, test_preds, test_trues = evaluate(
        model, test_loader, criterion, device, label_encoder, verbose=True
    )
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), 'spectral_multimodal_gnn.pth')
    with open('label_encoder_spectral.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('training_curves_spectral.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'train_accs': train_accs,
                     'val_losses': val_losses, 'val_accs': val_accs}, f)
    print("Saved: spectral_multimodal_gnn.pth, label_encoder_spectral.pkl, training_curves_spectral.pkl")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_range = range(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Spectral Multimodal GNN — Structure + Sequence + Spectral ({len(graphs)} proteins, best epoch {best_epoch})',
                     fontsize=12, fontweight='bold')
        ax1.plot(epochs_range, train_losses, label='Train loss', color='steelblue')
        ax1.plot(epochs_range, val_losses,   label='Val loss',   color='tomato', linestyle='--')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss')
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs_range, [a*100 for a in train_accs], label='Train acc', color='steelblue')
        ax2.plot(epochs_range, [a*100 for a in val_accs],   label='Val acc',   color='tomato', linestyle='--')
        ax2.axhline(y=test_acc*100, color='green', linestyle=':', label=f'Final Test Acc ({test_acc*100:.2f}%)')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy')
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves_spectral.png', dpi=150, bbox_inches='tight')
        print("Plot saved: training_curves_spectral.png")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
