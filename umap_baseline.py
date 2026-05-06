#!/usr/bin/env python3
"""
UMAP of Baseline Multimodal GNN learned embeddings.
Extracts [structure_out, sequence_out] features (512D) from the trained model,
runs UMAP, plots colored by class and taxa.
Output: umap_baseline.png
"""

import importlib_metadata, sys
sys.modules['importlib.metadata'] = importlib_metadata

import torch
import numpy as np
import pandas as pd
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

from multimodal_gnn_fixed import MultimodalToxinClassifier
from train_full_multimodal import (
    create_protein_graphs_batch,
    create_labels_from_metadata,
    load_fasta_based_data,
)

LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}
TAXA_LIST       = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']
CLASS_COLORS    = {'Calcium': '#e74c3c', 'Potassium': '#2ecc71', 'Sodium': '#3498db'}
TAXA_COLORS     = {'Snake': '#e67e22', 'Cone Snail': '#9b59b6', 'Spider': '#1abc9c',
                   'Scorpion': '#f39c12', 'Sea Anemone': '#e91e63'}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
esm_features, metadata_df = load_fasta_based_data()
labels, label_encoder, valid_proteins = create_labels_from_metadata(metadata_df)

taxa_lookup = dict(zip(
    pd.read_csv('taxa_fastas/taxa_metadata.csv')['protein_id'],
    pd.read_csv('taxa_fastas/taxa_metadata.csv')['taxa'],
))
taxa_labels = [taxa_lookup.get(p['protein_id']) for p in valid_proteins]

# ── Build graphs ──────────────────────────────────────────────────────────────
print("Building graphs...")
all_graphs = create_protein_graphs_batch(valid_proteins)

graphs, filt_labels, filt_taxa, filt_proteins = [], [], [], []
for i, g in enumerate(all_graphs):
    if g is not None and taxa_labels[i] is not None:
        graphs.append(g)
        filt_labels.append(labels[i])
        filt_taxa.append(taxa_labels[i])
        filt_proteins.append(valid_proteins[i])

esm_tensors   = torch.stack([torch.from_numpy(esm_features[p['feature_key']]).float()
                              for p in filt_proteins])
labels_tensor = torch.from_numpy(np.array(filt_labels)).long()
print(f"Proteins: {len(graphs)}")

# ── Dataset / loader ──────────────────────────────────────────────────────────
class SimpleDataset(Dataset):
    def __init__(self, graphs, esm, labels):
        self.graphs = graphs; self.esm = esm; self.labels = labels
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.esm[idx], self.labels[idx]

def collate_fn(batch):
    graphs, esm, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(labels)

loader = DataLoader(SimpleDataset(graphs, esm_tensors, labels_tensor),
                    batch_size=32, shuffle=False, collate_fn=collate_fn)

# ── Load model + extract embeddings ──────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = MultimodalToxinClassifier(
    esm_dim=1280, gnn_hidden_dim=256,
    num_classes=len(label_encoder.classes_),
    fusion_hidden_dim=512, dropout_rate=0.1,
).to(device)
model.load_state_dict(torch.load('multimodal_gnn_full_dataset.pth', map_location=device))
model.eval()

print("Extracting embeddings...")
all_feats, all_preds, all_labels = [], [], []
with torch.no_grad():
    for g, esm, y in loader:
        g, esm = g.to(device), esm.to(device)
        logits, structure_out, sequence_out = model(g, esm)
        combined = torch.cat([structure_out, sequence_out], dim=1)  # 512D
        all_feats.append(combined.cpu())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

features   = torch.cat(all_feats, dim=0).numpy()
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
acc = (all_preds == all_labels).mean()
print(f"Features shape: {features.shape} | Accuracy: {acc*100:.1f}%")

# ── UMAP ──────────────────────────────────────────────────────────────────────
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                    metric='euclidean', random_state=42, verbose=True)
coords  = reducer.fit_transform(features)
print("UMAP done")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Baseline Multimodal GNN — UMAP of learned embeddings (512D → 2D)\nAccuracy: {acc*100:.1f}%',
             fontsize=13, fontweight='bold')

class_names = label_encoder.classes_
for i, cls in enumerate(class_names):
    mask = all_labels == i
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=CLASS_COLORS.get(cls, '#999'), label=f'{cls} (n={mask.sum()})',
                alpha=0.7, s=20, edgecolors='none')
ax1.set_title('By Ion Channel Class', fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2'); ax1.grid(alpha=0.2)

for taxa in TAXA_LIST:
    mask = np.array(filt_taxa) == taxa
    if mask.sum() == 0: continue
    ax2.scatter(coords[mask, 0], coords[mask, 1],
                c=TAXA_COLORS.get(taxa, '#999'), label=f'{taxa} (n={mask.sum()})',
                alpha=0.7, s=20, edgecolors='none')
ax2.set_title('By Taxa', fontweight='bold')
ax2.legend(fontsize=9); ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2'); ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('umap_baseline.png', dpi=150, bbox_inches='tight')
print("Saved: umap_baseline.png")
