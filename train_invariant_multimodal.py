#!/usr/bin/env python3
"""
Invariant Multimodal GNN — SE(3)-invariant structural features
Instead of raw xyz coordinates (which change under rotation/translation),
node features are replaced with rotation-invariant local geometry:
  - Element one-hot (5D) — already invariant
  - Mean/std/min/max distance to spatial neighbors + neighbor count (5D)
  - Is disulfide bond participant (1D)
  Total: 11D invariant node features

Edge features: pairwise distance (1D) — passed to GATConv as edge_attr.
A protein's function doesn't change if you rotate it — this enforces that.
"""

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

from multimodal_gnn_fixed import SequenceEncoder

# ── Invariant structural encoder ──────────────────────────────────────────────

INV_NODE_DIM = 11  # 5 element + 5 distance stats + 1 disulfide flag

class InvariantStructuralEncoder(nn.Module):
    """
    SE(3)-invariant GNN: uses pairwise distances + local geometry instead of xyz.
    GATConv receives edge distances as edge_attr — attention is geometry-aware.
    """
    def __init__(self, node_features=INV_NODE_DIM, hidden_dim=256, num_layers=3, heads=4):
        super().__init__()
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # Spatial GATConv with edge distance features (edge_dim=1)
        self.spatial_convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)
            for _ in range(num_layers)
        ])

        # Disulfide attention (binary edges — no distance needed here)
        self.disulfide_attention = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index_spatial, edge_attr_spatial,
                edge_index_disulfide, batch):
        """
        x:                   [num_atoms, INV_NODE_DIM] invariant node features
        edge_index_spatial:  [2, num_spatial_edges]
        edge_attr_spatial:   [num_spatial_edges, 1] pairwise distances
        edge_index_disulfide:[2, num_disulfide_edges]
        batch:               [num_atoms]
        """
        h = F.relu(self.node_embedding(x))

        spatial_h = h
        for conv in self.spatial_convs:
            spatial_h = F.relu(conv(spatial_h, edge_index_spatial, edge_attr_spatial))
            spatial_h = F.dropout(spatial_h, p=0.2, training=self.training)

        if edge_index_disulfide.shape[1] > 0:
            disulfide_h = F.relu(self.disulfide_attention(h, edge_index_disulfide))
        else:
            disulfide_h = torch.zeros_like(spatial_h)

        combined = torch.cat([spatial_h, disulfide_h], dim=1)
        fused_h  = F.relu(self.fusion(combined))
        return global_mean_pool(fused_h, batch)


class InvariantMultimodalClassifier(nn.Module):
    def __init__(self, num_classes=4, esm_dim=1280, gnn_hidden_dim=256,
                 fusion_hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.structural_encoder = InvariantStructuralEncoder(hidden_dim=gnn_hidden_dim)
        self.sequence_encoder   = SequenceEncoder(esm_dim=esm_dim, hidden_dim=gnn_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, batch_graphs, esm_embeddings):
        structure_out = self.structural_encoder(
            batch_graphs.inv_x,
            batch_graphs.edge_index_spatial,
            batch_graphs.edge_attr_spatial,
            batch_graphs.edge_index_disulfide,
            batch_graphs.batch
        )
        sequence_out = self.sequence_encoder(esm_embeddings)
        combined = torch.cat([structure_out, sequence_out], dim=1)
        logits   = self.classifier(combined)
        return logits, structure_out, sequence_out

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

# ── Invariant graph construction ──────────────────────────────────────────────

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
    elem_feats = []
    for a in atoms_data:
        oh = [0] * 5
        oh[elem_map.get(a['element'], 0)] = 1
        elem_feats.append(oh)
    return torch.tensor(elem_feats, dtype=torch.float), coords, atoms_data


def detect_disulfide_bonds(coords, atoms_data, threshold=2.5):
    cys_sg = [i for i, a in enumerate(atoms_data)
              if a['residue_name'] == 'CYS' and a['atom_name'] == 'SG']
    bonds = []
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            if torch.norm(coords[cys_sg[i]] - coords[cys_sg[j]]).item() <= threshold:
                bonds.append((cys_sg[i], cys_sg[j]))
    return bonds


def build_spatial_edges_with_distances(coords, disulfide_pairs, threshold=4.0):
    """Returns edge_index and edge distances (invariant edge features)."""
    n = coords.shape[0]
    dsulf_set = set(disulfide_pairs) | {(j, i) for i, j in disulfide_pairs}
    edges, dists = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in dsulf_set:
                d = torch.norm(coords[i] - coords[j]).item()
                if d <= threshold:
                    edges.extend([(i, j), (j, i)])
                    dists.extend([d, d])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr  = torch.tensor(dists, dtype=torch.float).unsqueeze(1)  # [num_edges, 1]
    return edge_index, edge_attr


def build_disulfide_edges(bonds):
    if not bonds:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [(i, j) for i, j in bonds] + [(j, i) for i, j in bonds]
    return torch.tensor(edges, dtype=torch.long).t()


def compute_invariant_node_features(elem_feats, coords, edge_index_spatial,
                                     edge_attr_spatial, disulfide_pairs, num_atoms):
    """
    Builds 11D invariant node features:
      [element_onehot(5), mean_dist, std_dist, min_dist, max_dist, num_neighbors, is_disulfide]
    """
    dsulf_nodes = set()
    for i, j in disulfide_pairs:
        dsulf_nodes.add(i); dsulf_nodes.add(j)

    # Aggregate per-node distance stats from edge_attr
    mean_dist = torch.zeros(num_atoms)
    std_dist  = torch.zeros(num_atoms)
    min_dist  = torch.full((num_atoms,), float('inf'))
    max_dist  = torch.zeros(num_atoms)
    count     = torch.zeros(num_atoms)

    if edge_index_spatial.shape[1] > 0:
        src = edge_index_spatial[0]
        d   = edge_attr_spatial.squeeze(1)
        for idx in range(src.shape[0]):
            n_idx = src[idx].item()
            dist  = d[idx].item()
            count[n_idx]    += 1
            mean_dist[n_idx] += dist
            min_dist[n_idx]  = min(min_dist[n_idx].item(), dist)
            max_dist[n_idx]  = max(max_dist[n_idx].item(), dist)

        # Finalize mean
        nonzero = count > 0
        mean_dist[nonzero] /= count[nonzero]

        # Std: second pass
        for idx in range(src.shape[0]):
            n_idx = src[idx].item()
            std_dist[n_idx] += (d[idx].item() - mean_dist[n_idx].item()) ** 2
        std_dist[nonzero] = torch.sqrt(std_dist[nonzero] / count[nonzero])

        # Replace inf min with 0 for isolated nodes
        min_dist[min_dist == float('inf')] = 0.0

    is_disulfide = torch.tensor(
        [1.0 if i in dsulf_nodes else 0.0 for i in range(num_atoms)]
    )

    inv_x = torch.cat([
        elem_feats,                          # [N, 5]
        mean_dist.unsqueeze(1),              # [N, 1]
        std_dist.unsqueeze(1),               # [N, 1]
        min_dist.unsqueeze(1),               # [N, 1]
        max_dist.unsqueeze(1),               # [N, 1]
        count.unsqueeze(1),                  # [N, 1]
        is_disulfide.unsqueeze(1),           # [N, 1]
    ], dim=1)  # [N, 11]

    return inv_x


def create_protein_graph(protein_info):
    try:
        elem_feats, coords, atoms = parse_pdb_to_tensors(protein_info['structure_file'])
        dsulf      = detect_disulfide_bonds(coords, atoms)
        ei_spatial, ea_spatial = build_spatial_edges_with_distances(coords, dsulf)
        ei_disulf  = build_disulfide_edges(dsulf)
        inv_x      = compute_invariant_node_features(
            elem_feats, coords, ei_spatial, ea_spatial, dsulf, coords.shape[0]
        )
        return Data(
            inv_x=inv_x,
            edge_index_spatial=ei_spatial,
            edge_attr_spatial=ea_spatial,
            edge_index_disulfide=ei_disulf,
            num_nodes=coords.shape[0],
        )
    except Exception as e:
        print(f"Error {protein_info['protein_id']}: {e}")
        return None

# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (g, esm, y) in enumerate(loader):
        g, esm, y = g.to(device), esm.to(device), y.to(device)
        optimizer.zero_grad()
        logits, *_ = model(g, esm)
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
        for g, esm, y in loader:
            g, esm, y = g.to(device), esm.to(device), y.to(device)
            logits, *_ = model(g, esm)
            total_loss += criterion(logits, y).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(trues)).mean()
    if verbose:
        print(classification_report(trues, preds, target_names=label_encoder.classes_))
    return total_loss / len(loader), acc, preds, trues

# ── Dataset / collate ─────────────────────────────────────────────────────────

class InvariantDataset(torch.utils.data.Dataset):
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
    print("=== Invariant Multimodal GNN (SE(3)-invariant Structure + ESM Sequence) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    esm_features, metadata_df = load_fasta_based_data()
    labels, label_encoder, valid_proteins = create_labels_from_metadata(metadata_df)
    print(f"Dataset: {len(valid_proteins)} proteins")

    print("=== Building Invariant Protein Graphs ===")
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
    print(f"Node feature dim: {graphs[0].inv_x.shape[1]}D (invariant)")

    esm_tensors   = torch.stack([
        torch.from_numpy(esm_features[p['feature_key']]).float()
        for p in filtered_proteins
    ])
    labels_tensor = torch.from_numpy(np.array(filtered_labels)).long()
    print(f"ESM: {esm_tensors.shape}, Labels: {labels_tensor.shape}")

    train_idx, test_idx = train_test_split(
        range(len(graphs)), test_size=0.2, random_state=42, stratify=filtered_labels
    )

    train_ds = InvariantDataset(
        [graphs[i] for i in train_idx], esm_tensors[train_idx], labels_tensor[train_idx]
    )
    test_ds = InvariantDataset(
        [graphs[i] for i in test_idx], esm_tensors[test_idx], labels_tensor[test_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    num_classes = len(label_encoder.classes_)
    model = InvariantMultimodalClassifier(
        num_classes=num_classes,
        esm_dim=1280,
        gnn_hidden_dim=256,
        fusion_hidden_dim=512,
        dropout_rate=0.1
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Class-weighted loss — Calcium outnumbered ~3:1 by K/Na
    train_labels_tensor = labels_tensor[list(train_idx)]
    class_counts  = torch.bincount(train_labels_tensor)
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
            torch.save(model.state_dict(), 'invariant_multimodal_gnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    print(f"\nTotal training time: {time.time()-t0:.1f}s")
    model.load_state_dict(torch.load('invariant_multimodal_gnn.pth'))
    print(f"Loaded best model from epoch {best_epoch}")

    print("\n=== Final Evaluation ===")
    test_loss, test_acc, test_preds, test_trues = evaluate(
        model, test_loader, criterion, device, label_encoder, verbose=True
    )
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")
    with open('label_encoder_invariant.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('training_curves_invariant.pkl', 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'train_accs': train_accs,
                     'val_losses': val_losses, 'val_accs': val_accs}, f)
    print("Saved: invariant_multimodal_gnn.pth, label_encoder_invariant.pkl, training_curves_invariant.pkl")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Invariant Multimodal GNN — SE(3)-invariant Structure + ESM ({len(graphs)} proteins, {num_epochs} epochs)',
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
        plt.savefig('training_curves_invariant.png', dpi=150, bbox_inches='tight')
        print("Plot saved: training_curves_invariant.png")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
