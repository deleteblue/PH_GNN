#!/usr/bin/env python3
"""
t-SNE of Homology GNN learned embeddings.
Extracts the concatenated [structure, sequence, homology] features (768D)
from the trained model, runs t-SNE, plots colored by class and taxa.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_homology_multimodal import (
    HomologyMultimodalClassifier, HOMOLOGY_DIM,
    create_protein_graph, compute_homology_features_batch,
    HomologyDataset, collate_fn,
)

LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}
TAXA_LIST       = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
esm_features = torch.load('esm_features_fasta_aligned.pt')
metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
cone_df     = pd.read_excel('cone_trainingData.xlsx')
toxin_df     = pd.read_excel('toxinCombined_Finalv3.xlsx')
taxa_lookup  = dict(zip(
    pd.read_csv('taxa_fastas/taxa_metadata.csv')['protein_id'],
    pd.read_csv('taxa_fastas/taxa_metadata.csv')['taxa'],
))

labels, proteins, taxa_labels, protein_ids = [], [], [], []
for _, row in metadata_df.iterrows():
    pid, dataset = row['protein_id'], row['dataset']
    if dataset == 'cone':
        pr = cone_df[cone_df['Entry'] == pid]
        if len(pr) == 0: continue
        r = pr.iloc[0]
        if r['match'] == r['Zhang result']:    cls = r['match']
        elif r['match'] == 'Other':            cls = r['Zhang result']
        elif r['Zhang result'] == 'Other':     cls = r['match']
        else:                                  continue
        cls = LABEL_NORMALIZE.get(cls, cls)
    elif dataset == 'toxin_combined':
        pr = toxin_df[toxin_df['entry'] == pid]
        if len(pr) == 0: continue
        cls = pr.iloc[0]['ionChannelClass']
    else:
        continue
    taxa = taxa_lookup.get(pid)
    if taxa is None: continue
    labels.append(cls); proteins.append(row)
    taxa_labels.append(taxa); protein_ids.append(pid)

unique_classes = sorted(set(labels))
class_to_idx   = {c: i for i, c in enumerate(unique_classes)}
labels_idx     = np.array([class_to_idx[l] for l in labels])
print(f"Proteins: {len(proteins)} | Classes: {unique_classes}")

# ── Build graphs + homology features ─────────────────────────────────────────
print("Building graphs...")
graphs, filt_labels, filt_taxa, filt_ids = [], [], [], []
filt_proteins = []
for i, p in enumerate(proteins):
    if (i+1) % 200 == 0: print(f"  {i+1}/{len(proteins)}")
    g = create_protein_graph(p)
    if g is not None:
        graphs.append(g)
        filt_proteins.append(p)
        filt_labels.append(labels_idx[i])
        filt_taxa.append(taxa_labels[i])
        filt_ids.append(protein_ids[i])

print("Computing homology features...")
hom_tensors  = compute_homology_features_batch(filt_proteins)
esm_tensors  = torch.stack([torch.from_numpy(esm_features[p['feature_key']]).float() for p in filt_proteins])
labels_tensor = torch.from_numpy(np.array(filt_labels)).long()

loader = DataLoader(
    HomologyDataset(graphs, esm_tensors, hom_tensors, labels_tensor),
    batch_size=32, shuffle=False, collate_fn=collate_fn,
)

# ── Load model + extract embeddings ──────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = HomologyMultimodalClassifier(
    num_classes=len(unique_classes), esm_dim=1280,
    gnn_hidden_dim=256, homology_dim=HOMOLOGY_DIM,
    fusion_hidden_dim=512, dropout_rate=0.1,
).to(device)
model.load_state_dict(torch.load('homology_multimodal_gnn.pth', map_location=device))
model.eval()

print("Extracting embeddings...")
all_feats, all_preds, all_labels = [], [], []
with torch.no_grad():
    for g, esm, hom, y in loader:
        g, esm, hom = g.to(device), esm.to(device), hom.to(device)
        logits, structure_out, sequence_out, homology_out = model(g, esm, hom)
        # Concatenate all three modality outputs → 768D representation
        combined = torch.cat([structure_out, sequence_out, homology_out], dim=1)
        all_feats.append(combined.cpu())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

features   = torch.cat(all_feats, dim=0).numpy()
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
acc = (all_preds == all_labels).mean()
print(f"Features shape: {features.shape} | Accuracy: {acc*100:.1f}%")

# ── PCA → t-SNE ───────────────────────────────────────────────────────────────
print("Running PCA → t-SNE...")
t0      = time.time()
reduced = PCA(n_components=50, random_state=42).fit_transform(features)
coords  = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42).fit_transform(reduced)
print(f"t-SNE done in {time.time()-t0:.0f}s")

# ── Plot ──────────────────────────────────────────────────────────────────────
CLASS_COLORS = {'Calcium': '#e74c3c', 'Potassium': '#2ecc71', 'Sodium': '#3498db'}
TAXA_COLORS  = {'Snake': '#e67e22', 'Cone Snail': '#9b59b6', 'Spider': '#1abc9c',
                'Scorpion': '#f39c12', 'Sea Anemone': '#e91e63'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Homology GNN — t-SNE of learned embeddings (768D → 2D)\nAccuracy: {acc*100:.1f}%',
             fontsize=13, fontweight='bold')

for cls, idx in class_to_idx.items():
    mask = all_labels == idx
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=CLASS_COLORS.get(cls, '#999'), label=f'{cls} (n={mask.sum()})',
                alpha=0.7, s=20, edgecolors='none')
ax1.set_title('Colored by Ion Channel Class', fontweight='bold')
ax1.legend(fontsize=9); ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2'); ax1.grid(alpha=0.2)

for taxa in TAXA_LIST:
    mask = np.array(filt_taxa) == taxa
    if mask.sum() == 0: continue
    ax2.scatter(coords[mask, 0], coords[mask, 1],
                c=TAXA_COLORS.get(taxa, '#999'), label=f'{taxa} (n={mask.sum()})',
                alpha=0.7, s=20, edgecolors='none')
ax2.set_title('Colored by Taxa', fontweight='bold')
ax2.legend(fontsize=9); ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('tsne_homology.png', dpi=150, bbox_inches='tight')
print("Saved: tsne_homology.png")
