#!/usr/bin/env python3
"""
t-SNE and UMAP of learned embeddings for ALL 6 multimodal GNN models.
Extracts pre-classifier fused embeddings, plots colored by ion class and taxa.
Output: tsne_umap_{model}.png  for each model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spectral_agent'))

import importlib_metadata
sys.modules['importlib.metadata'] = importlib_metadata

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Consistent color palettes (no similar yellows) ───────────────────────────

TAXA_COLORS = {
    'Snake':       '#e41a1c',   # vivid red
    'Cone Snail':  '#377eb8',   # steel blue
    'Spider':      '#4daf4a',   # forest green
    'Scorpion':    '#ff7f00',   # vivid orange
    'Sea Anemone': '#984ea3',   # purple
}
TAXA_MARKERS = {
    'Snake': 'o', 'Cone Snail': 's', 'Spider': '^', 'Scorpion': 'D', 'Sea Anemone': '*'
}

CLASS_COLORS = {
    'Calcium':   '#1f77b4',   # blue
    'Potassium': '#2ca02c',   # green
    'Sodium':    '#d62728',   # red
}
CLASS_MARKERS = {'Calcium': 'o', 'Potassium': 's', 'Sodium': '^'}

TAXA_LIST = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']
LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}

# ── Common data loading ───────────────────────────────────────────────────────

def load_common_data():
    print("=== Loading common data ===")
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
    cone_df     = pd.read_excel('cone_trainingData.xlsx')
    toxin_df     = pd.read_excel('toxinCombined_Finalv3.xlsx')
    taxa_lookup  = dict(zip(
        pd.read_csv('taxa_fastas/taxa_metadata.csv')['protein_id'],
        pd.read_csv('taxa_fastas/taxa_metadata.csv')['taxa'],
    ))

    labels, proteins, taxa_labels = [], [], []
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
        labels.append(cls); proteins.append(row); taxa_labels.append(taxa)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded = label_encoder.transform(labels)
    print(f"Loaded {len(proteins)} proteins | Classes: {label_encoder.classes_}")
    return esm_features, proteins, encoded, label_encoder, taxa_labels


# ── Graph builders ────────────────────────────────────────────────────────────

def build_standard_graphs(proteins):
    from train_full_multimodal import (
        parse_pdb_to_tensors, detect_disulfide_bonds,
        build_spatial_edges, build_disulfide_edges,
    )
    from torch_geometric.data import Data
    print("Building standard graphs...")
    graphs, keep_idx = [], []
    for i, p in enumerate(proteins):
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(proteins)}")
        try:
            node_feats, coords, atoms = parse_pdb_to_tensors(p['structure_file'])
            dsulf = detect_disulfide_bonds(coords, atoms)
            graphs.append(Data(
                x=node_feats, pos=coords,
                edge_index_spatial=build_spatial_edges(coords, dsulf),
                edge_index_disulfide=build_disulfide_edges(dsulf),
            ))
            keep_idx.append(i)
        except Exception as e:
            print(f"  Skip {p['protein_id']}: {e}")
    print(f"  Built {len(graphs)} graphs")
    return graphs, keep_idx


def build_invariant_graphs(proteins):
    from train_invariant_multimodal import create_protein_graph
    print("Building invariant graphs...")
    graphs, keep_idx = [], []
    for i, p in enumerate(proteins):
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(proteins)}")
        g = create_protein_graph(p)
        if g is not None:
            graphs.append(g); keep_idx.append(i)
    print(f"  Built {len(graphs)} graphs")
    return graphs, keep_idx


# ── Auxiliary feature builders ────────────────────────────────────────────────

def build_homology_features(proteins):
    from train_homology_multimodal import compute_homology_features_batch
    print("Computing homology features...")
    return compute_homology_features_batch(proteins)


def build_spectral_features(graphs, proteins):
    from train_spectral_multimodal import compute_spectral_features_batch
    print("Computing spectral features...")
    return compute_spectral_features_batch(graphs, proteins)


# ── Dataset / collate helpers ─────────────────────────────────────────────────

class SimpleDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def std_collate(batch):
    graphs, esm, y = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(y)

def tri_collate(batch):
    graphs, esm, aux, y = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(aux), torch.stack(y)


# ── Embedding extractors ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_std(model, graphs, esm_tensors, labels_tensor, device, batch_size=32):
    """Baseline / Riemannian / Invariant — returns (logits, cat[struct, seq])"""
    loader = DataLoader(
        SimpleDataset(list(zip(graphs, esm_tensors, labels_tensor))),
        batch_size=batch_size, collate_fn=std_collate
    )
    feats, preds, trues = [], [], []
    model.eval()
    for g, esm, y in loader:
        g, esm = g.to(device), esm.to(device)
        logits, struct, seq = model(g, esm)
        feats.append(torch.cat([struct, seq], dim=1).cpu())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.concatenate([f.numpy() for f in feats]), np.array(preds), np.array(trues)


@torch.no_grad()
def extract_manifold(model, graphs, esm_tensors, labels_tensor, device, batch_size=32):
    """Statistical Manifold — uses precision_weighted from sequence encoder"""
    from train_full_multimodal import (
        parse_pdb_to_tensors, detect_disulfide_bonds,
        build_spatial_edges, build_disulfide_edges,
    )
    loader = DataLoader(
        SimpleDataset(list(zip(graphs, esm_tensors, labels_tensor))),
        batch_size=batch_size, collate_fn=std_collate
    )
    feats, preds, trues = [], [], []
    model.eval()
    for g, esm, y in loader:
        g, esm = g.to(device), esm.to(device)
        struct = model.structural_encoder(
            g.x, g.pos, g.edge_index_spatial, g.edge_index_disulfide, g.batch)
        mu, log_var, precision_weighted = model.sequence_encoder(esm)
        combined = torch.cat([precision_weighted, struct], dim=1)
        logits = model.classifier(combined)
        feats.append(combined.cpu())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.concatenate([f.numpy() for f in feats]), np.array(preds), np.array(trues)


@torch.no_grad()
def extract_homology(model, graphs, esm_tensors, hom_tensors, labels_tensor, device, batch_size=32):
    loader = DataLoader(
        SimpleDataset(list(zip(graphs, esm_tensors, hom_tensors, labels_tensor))),
        batch_size=batch_size, collate_fn=tri_collate
    )
    feats, preds, trues = [], [], []
    model.eval()
    for g, esm, hom, y in loader:
        g, esm, hom = g.to(device), esm.to(device), hom.to(device)
        logits, struct, seq, hom_out = model(g, esm, hom)
        feats.append(torch.cat([struct, seq, hom_out], dim=1).cpu())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.concatenate([f.numpy() for f in feats]), np.array(preds), np.array(trues)


@torch.no_grad()
def extract_spectral(model, graphs, esm_tensors, spec_tensors, labels_tensor, device, batch_size=32):
    loader = DataLoader(
        SimpleDataset(list(zip(graphs, esm_tensors, spec_tensors, labels_tensor))),
        batch_size=batch_size, collate_fn=tri_collate
    )
    feats, preds, trues = [], [], []
    model.eval()
    for g, esm, spec, y in loader:
        g, esm, spec = g.to(device), esm.to(device), spec.to(device)
        logits, struct, seq, spec_out = model(g, esm, spec)
        feats.append(torch.cat([struct, seq, spec_out], dim=1).cpu())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.concatenate([f.numpy() for f in feats]), np.array(preds), np.array(trues)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def run_tsne(features):
    print(f"  PCA(50) → t-SNE ...")
    t0 = time.time()
    reduced = PCA(n_components=min(50, features.shape[1]), random_state=42).fit_transform(features)
    coords  = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42).fit_transform(reduced)
    print(f"  t-SNE done in {time.time()-t0:.0f}s")
    return coords


def run_umap(features):
    print(f"  UMAP ...")
    t0 = time.time()
    coords = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(features)
    print(f"  UMAP done in {time.time()-t0:.0f}s")
    return coords


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plot(tsne_coords, umap_coords, labels, taxa, label_encoder, acc, model_name, out_path):
    classes   = list(label_encoder.classes_)
    taxa_list = [t for t in TAXA_LIST if t in set(taxa)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} — t-SNE & UMAP  (Accuracy: {acc*100:.1f}%)',
                 fontsize=14, fontweight='bold')

    titles = [('t-SNE', 'Colored by Ion Channel Class'),
              ('t-SNE', 'Colored by Taxa'),
              ('UMAP',  'Colored by Ion Channel Class'),
              ('UMAP',  'Colored by Taxa')]
    coords_list = [tsne_coords, tsne_coords, umap_coords, umap_coords]
    color_modes = ['class', 'taxa', 'class', 'taxa']

    for ax, (method, subtitle), coords, mode in zip(axes.flat, titles, coords_list, color_modes):
        if mode == 'class':
            for ci, cls in enumerate(classes):
                mask = labels == ci
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                           s=18, alpha=0.7, linewidths=0, label=f'{cls} (n={mask.sum()})')
        else:
            for taxa_name in taxa_list:
                mask = np.array(taxa) == taxa_name
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=TAXA_COLORS[taxa_name], marker=TAXA_MARKERS[taxa_name],
                           s=18, alpha=0.7, linewidths=0, label=f'{taxa_name} (n={mask.sum()})')

        ax.set_title(f'{method} — {subtitle}', fontsize=10, fontweight='bold')
        ax.set_xlabel(f'{method} 1'); ax.set_ylabel(f'{method} 2')
        ax.legend(fontsize=7, markerscale=1.5, framealpha=0.8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    esm_features, proteins, labels, label_encoder, taxa_labels = load_common_data()
    labels_tensor = torch.from_numpy(labels).long()

    # ── Build graphs (standard + invariant) ──────────────────────────────────
    std_graphs, std_idx = build_standard_graphs(proteins)
    inv_graphs, inv_idx = build_invariant_graphs(proteins)

    std_esm     = torch.stack([torch.from_numpy(esm_features[proteins[i]['feature_key']]).float() for i in std_idx])
    std_labels  = labels_tensor[std_idx]
    std_taxa    = [taxa_labels[i] for i in std_idx]

    inv_esm     = torch.stack([torch.from_numpy(esm_features[proteins[i]['feature_key']]).float() for i in inv_idx])
    inv_labels  = labels_tensor[inv_idx]
    inv_taxa    = [taxa_labels[i] for i in inv_idx]

    std_proteins = [proteins[i] for i in std_idx]

    # ── Homology & spectral features (computed once for std proteins) ─────────
    hom_tensors  = build_homology_features(std_proteins)
    spec_tensors = build_spectral_features(std_graphs, std_proteins)

    # ── Model configs ─────────────────────────────────────────────────────────
    # Each entry: (name, weights_file, setup_fn)
    def run_baseline():
        from multimodal_gnn_fixed import MultimodalToxinClassifier
        model = MultimodalToxinClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('multimodal_gnn_full_dataset.pth', map_location=device))
        feats, preds, trues = extract_std(model, std_graphs, std_esm, std_labels, device)
        return feats, preds, trues, std_taxa

    def run_riemannian():
        from train_riemannian_multimodal import RiemannianToxinClassifier
        model = RiemannianToxinClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('riemannian_multimodal_gnn.pth', map_location=device))
        feats, preds, trues = extract_std(model, std_graphs, std_esm, std_labels, device)
        return feats, preds, trues, std_taxa

    def run_manifold():
        from train_statistical_manifold_multimodal import StatisticalManifoldClassifier
        model = StatisticalManifoldClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('statistical_manifold_gnn.pth', map_location=device))
        feats, preds, trues = extract_manifold(model, std_graphs, std_esm, std_labels, device)
        return feats, preds, trues, std_taxa

    def run_invariant():
        from train_invariant_multimodal import InvariantMultimodalClassifier
        model = InvariantMultimodalClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('invariant_multimodal_gnn.pth', map_location=device))
        feats, preds, trues = extract_std(model, inv_graphs, inv_esm, inv_labels, device)
        return feats, preds, trues, inv_taxa

    def run_homology():
        from train_homology_multimodal import HomologyMultimodalClassifier, HOMOLOGY_DIM
        model = HomologyMultimodalClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, homology_dim=HOMOLOGY_DIM,
            fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('homology_multimodal_gnn.pth', map_location=device))
        feats, preds, trues = extract_homology(model, std_graphs, std_esm, hom_tensors, std_labels, device)
        return feats, preds, trues, std_taxa

    def run_spectral():
        from train_spectral_multimodal import SpectralMultimodalClassifier
        model = SpectralMultimodalClassifier(num_classes=3, esm_dim=1280,
            gnn_hidden_dim=256, spectral_dim=16,
            fusion_hidden_dim=512, dropout_rate=0.1).to(device)
        model.load_state_dict(torch.load('spectral_multimodal_gnn.pth', map_location=device))
        feats, preds, trues = extract_spectral(model, std_graphs, std_esm, spec_tensors, std_labels, device)
        return feats, preds, trues, std_taxa

    model_runners = [
        ('Baseline GNN',           run_baseline),
        ('Riemannian GNN',         run_riemannian),
        ('Statistical Manifold',   run_manifold),
        ('Invariant SE(3) GNN',    run_invariant),
        ('Homology GNN',           run_homology),
        ('Spectral GNN',           run_spectral),
    ]

    for model_name, runner in model_runners:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        try:
            feats, preds, trues, taxa = runner()
            acc = (preds == trues).mean()
            print(f"  Embeddings: {feats.shape}  Accuracy: {acc*100:.1f}%")

            tsne_coords = run_tsne(feats)
            umap_coords = run_umap(feats)

            safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            out_path  = f'tsne_umap_{safe_name}.png'
            make_plot(tsne_coords, umap_coords, trues, taxa,
                      label_encoder, acc, model_name, out_path)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\n=== All done ===")


if __name__ == '__main__':
    main()
