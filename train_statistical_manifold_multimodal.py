#!/usr/bin/env python3
"""
Statistical Manifold Multimodal GNN — Structure + Fisher Information Sequence
Each protein's ESM embedding is modeled as a Gaussian distribution N(μ, σ²) on
the protein manifold. The Fisher information (precision = 1/σ²) weights the
sequence features before fusion: certain dimensions are amplified, uncertain ones
are dampened. Loss = CrossEntropy + β * KL(N(μ,σ²) || N(0,I)).

Key difference from Riemannian: outputs TWO vectors (μ, σ) instead of one,
and explicitly captures prediction uncertainty per protein.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pickle
import time

from multimodal_gnn_fixed import StructuralEncoder

# ── Probabilistic sequence encoder ───────────────────────────────────────────

class ProbabilisticSequenceEncoder(nn.Module):
    """
    Encodes ESM embeddings as a Gaussian distribution N(μ, σ²).
    precision = 1/σ² is the Fisher information — it measures how
    informative each hidden dimension is for classification.
    """
    def __init__(self, esm_dim=1280, hidden_dim=256):
        super().__init__()
        # Shared projection
        self.shared = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Mean head
        self.mu_head     = nn.Linear(hidden_dim * 2, hidden_dim)
        # Log-variance head (predict log σ² for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, esm_embeddings):
        """
        esm_embeddings: [batch_size, esm_dim]
        Returns:
            mu:      [batch, hidden_dim] — mean on protein manifold
            log_var: [batch, hidden_dim] — log variance (uncertainty)
            precision_weighted: [batch, hidden_dim] — mu * (1/σ), Fisher-weighted features
        """
        h       = self.shared(esm_embeddings)
        mu      = self.mu_head(h)
        log_var = torch.clamp(self.logvar_head(h), min=-10, max=4)  # Numerical stability

        # Precision = 1/σ = exp(-0.5 * log_var) — this IS the Fisher information
        precision           = torch.exp(-0.5 * log_var)
        precision_weighted  = mu * precision  # High-certainty dims amplified

        return mu, log_var, precision_weighted


class StatisticalManifoldClassifier(nn.Module):
    def __init__(self, num_classes=4, esm_dim=1280, gnn_hidden_dim=256,
                 fusion_hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.structural_encoder = StructuralEncoder(hidden_dim=gnn_hidden_dim)
        self.sequence_encoder   = ProbabilisticSequenceEncoder(
            esm_dim=esm_dim, hidden_dim=gnn_hidden_dim
        )
        # Fuse precision-weighted sequence (256) + structure (256) = 512
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, batch_graphs, esm_embeddings):
        structure_out = self.structural_encoder(
            batch_graphs.x,
            batch_graphs.pos,
            batch_graphs.edge_index_spatial,
            batch_graphs.edge_index_disulfide,
            batch_graphs.batch
        )
        mu, log_var, precision_weighted = self.sequence_encoder(esm_embeddings)

        combined = torch.cat([precision_weighted, structure_out], dim=1)
        logits   = self.classifier(combined)
        return logits, mu, log_var, structure_out


def kl_divergence_loss(mu, log_var):
    """KL(N(μ,σ²) || N(0,I)) — regularizes distributions toward unit Gaussian."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

# ── Data loading (identical across all scripts) ───────────────────────────────

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
    label_encoder  = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")
    print(f"Classes: {label_encoder.classes_}")
    return encoded_labels, label_encoder, valid_proteins

# ── Graph construction (identical across all scripts) ─────────────────────────

def parse_pdb_to_tensors(pdb_path):
    atoms_data = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms_data.append({
                    'atom_name':    line[12:16].strip(),
                    'residue_name': line[17:20].strip(),
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

# ── Training / evaluation ─────────────────────────────────────────────────────

KL_BETA = 0.01  # Weight of KL term — small so it doesn't dominate classification loss

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (g, esm, y) in enumerate(loader):
        g, esm, y = g.to(device), esm.to(device), y.to(device)
        optimizer.zero_grad()
        logits, mu, log_var, _ = model(g, esm)
        ce_loss = criterion(logits, y)
        kl_loss = kl_divergence_loss(mu, log_var)
        loss    = ce_loss + KL_BETA * kl_loss
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  ce={ce_loss.item():.4f}  kl={kl_loss.item():.4f}")
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device, label_encoder, verbose=False):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for g, esm, y in loader:
            g, esm, y = g.to(device), esm.to(device), y.to(device)
            logits, mu, log_var, _ = model(g, esm)
            ce_loss = criterion(logits, y)
            kl_loss = kl_divergence_loss(mu, log_var)
            total_loss += (ce_loss + KL_BETA * kl_loss).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(trues)).mean()
    if verbose:
        print(classification_report(trues, preds, target_names=label_encoder.classes_))
    return total_loss / len(loader), acc, preds, trues

# ── Dataset / collate ─────────────────────────────────────────────────────────

class ManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, esm, labels):
        self.graphs = graphs
        self.esm    = esm
        self.labels = labels

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.esm[idx], self.labels[idx]


def collate_fn(batch):
    graphs, esm, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(labels)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Statistical Manifold Multimodal GNN (Structure + Fisher Information Sequence) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"KL beta: {KL_BETA}")

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

    esm_tensors   = torch.stack([
        torch.from_numpy(esm_features[p['feature_key']]).float()
        for p in filtered_proteins
    ])
    labels_tensor = torch.from_numpy(np.array(filtered_labels)).long()
    print(f"ESM: {esm_tensors.shape}, Labels: {labels_tensor.shape}")

    # Train / test split
    train_idx, test_idx = train_test_split(
        range(len(graphs)), test_size=0.2, random_state=42, stratify=filtered_labels
    )

    train_ds = ManifoldDataset(
        [graphs[i] for i in train_idx], esm_tensors[train_idx], labels_tensor[train_idx]
    )
    test_ds = ManifoldDataset(
        [graphs[i] for i in test_idx], esm_tensors[test_idx], labels_tensor[test_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    num_classes = len(label_encoder.classes_)
    model = StatisticalManifoldClassifier(
        num_classes=num_classes,
        esm_dim=1280,
        gnn_hidden_dim=256,
        fusion_hidden_dim=512,
        dropout_rate=0.1
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_counts  = torch.bincount(labels_tensor[list(train_idx)])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))

    num_epochs       = 50
    patience         = 10
    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    print(f"\n=== Training up to {num_epochs} epochs (early stopping patience={patience}) ===")
    t0 = time.time()

    for epoch in range(num_epochs):
        t_ep = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, device, label_encoder)
        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(vl_loss);   val_accs.append(vl_acc)
        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}  "
              f"({time.time()-t_ep:.1f}s)")
        if vl_loss < best_val_loss - 1e-4:
            best_val_loss    = vl_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), 'statistical_manifold_gnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    model.load_state_dict(torch.load('statistical_manifold_gnn.pth'))
    print(f"Loaded best model from epoch {best_epoch}")

    print("\n=== Final Evaluation ===")
    test_loss, test_acc, test_preds, test_trues = evaluate(
        model, test_loader, criterion, device, label_encoder, verbose=True
    )
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
    with open('label_encoder_manifold.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('training_curves_manifold.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'train_accs': train_accs,
                     'val_losses': val_losses, 'val_accs': val_accs}, f)
    print("Saved: statistical_manifold_gnn.pth, label_encoder_manifold.pkl, training_curves_manifold.pkl")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_range = range(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Statistical Manifold GNN — Structure + Fisher Sequence ({len(graphs)} proteins, {num_epochs} epochs)',
                     fontsize=12, fontweight='bold')
        ax1.plot(epochs_range, train_losses, label='Train loss', color='steelblue')
        ax1.plot(epochs_range, val_losses,   label='Val loss',   color='tomato', linestyle='--')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (CE + KL)'); ax1.set_title('Loss')
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs_range, [a*100 for a in train_accs], label='Train acc', color='steelblue')
        ax2.plot(epochs_range, [a*100 for a in val_accs],   label='Val acc',   color='tomato', linestyle='--')
        ax2.axhline(y=test_acc*100, color='green', linestyle=':', label=f'Final Test Acc ({test_acc*100:.2f}%)')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy')
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves_manifold.png', dpi=150, bbox_inches='tight')
        print("Plot saved: training_curves_manifold.png")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
