#!/usr/bin/env python3
"""
XAI Embedding Visualization — all Multimodal GNN variants.
Usage:
    python visualize_embeddings.py --model homology
    python visualize_embeddings.py --model baseline
    python visualize_embeddings.py --model riemannian
    python visualize_embeddings.py --model invariant
    python visualize_embeddings.py --model manifold

For each model:
  - Loads saved checkpoint
  - Extracts 512D fusion embeddings via forward hook on classifier ReLU
  - Runs PCA (→50D) then t-SNE (→2D)
  - Saves static PNG (3 panels: class / taxa / correct-wrong)
  - Saves interactive Plotly HTML (hover = protein_id, taxa, true, pred, confidence)

All outputs prefixed with model name — no overwriting between models.
Embeddings cached to disk so t-SNE can be rerun without re-inferring.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}
TAXA_LIST       = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']

MODEL_CONFIGS = {
    'homology':   {'ckpt': 'homology_multimodal_gnn.pth',         'label_pkl': 'label_encoder_homology.pkl'},
    'baseline':   {'ckpt': 'multimodal_gnn_full_dataset.pth',     'label_pkl': 'label_encoder_full_dataset.pkl'},
    'riemannian': {'ckpt': 'riemannian_multimodal_gnn.pth',       'label_pkl': 'label_encoder_riemannian.pkl'},
    'invariant':  {'ckpt': 'invariant_multimodal_gnn.pth',        'label_pkl': 'label_encoder_invariant.pkl'},
    'manifold':   {'ckpt': 'statistical_manifold_gnn.pth',        'label_pkl': 'label_encoder_manifold.pkl'},
}

# ── Shared data loading ───────────────────────────────────────────────────────

def load_base_data():
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
    return esm_features, metadata_df


def load_all_metadata(metadata_df):
    """Returns proteins with taxa + class labels."""
    cone_df    = pd.read_excel('cone_trainingData.xlsx')
    toxin_df    = pd.read_excel('toxinCombined_Finalv3.xlsx')
    taxa_df     = pd.read_csv('taxa_fastas/taxa_metadata.csv')[['protein_id', 'taxa']]
    taxa_lookup = dict(zip(taxa_df['protein_id'], taxa_df['taxa']))

    labels, valid_proteins, taxa_labels, protein_ids = [], [], [], []
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
        labels.append(cls); valid_proteins.append(row)
        taxa_labels.append(taxa); protein_ids.append(pid)

    label_encoder  = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder, valid_proteins, taxa_labels, protein_ids

# ── Model-specific data pipelines ────────────────────────────────────────────

def build_dataset_homology(valid_proteins, esm_features, labels):
    from train_homology_multimodal import (
        create_protein_graph, compute_homology_features_batch,
        HomologyDataset, collate_fn,
    )
    graphs, filt_p, filt_l = [], [], []
    for i, p in enumerate(valid_proteins):
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(valid_proteins)}")
        g = create_protein_graph(p)
        if g is not None:
            graphs.append(g); filt_p.append(p); filt_l.append(labels[i])
    hom   = compute_homology_features_batch(filt_p)
    esm   = torch.stack([torch.from_numpy(esm_features[p['feature_key']]).float() for p in filt_p])
    lbls  = torch.from_numpy(np.array(filt_l)).long()
    ds    = HomologyDataset(graphs, esm, hom, lbls)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return loader, filt_l, filt_p


def build_dataset_standard(valid_proteins, esm_features, labels, model_name):
    """Shared pipeline for baseline, riemannian, manifold."""
    if model_name == 'invariant':
        from train_invariant_multimodal import create_protein_graph
    else:
        from train_homology_multimodal import create_protein_graph

    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, graphs, esm, labels):
            self.graphs = graphs; self.esm = esm; self.labels = labels
        def __len__(self): return len(self.graphs)
        def __getitem__(self, idx): return self.graphs[idx], self.esm[idx], self.labels[idx]

    def collate_fn(batch):
        graphs, esm, labels = zip(*batch)
        return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(labels)

    graphs, filt_p, filt_l = [], [], []
    for i, p in enumerate(valid_proteins):
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(valid_proteins)}")
        g = create_protein_graph(p)
        if g is not None:
            graphs.append(g); filt_p.append(p); filt_l.append(labels[i])
    esm  = torch.stack([torch.from_numpy(esm_features[p['feature_key']]).float() for p in filt_p])
    lbls = torch.from_numpy(np.array(filt_l)).long()
    ds   = SimpleDataset(graphs, esm, lbls)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return loader, filt_l, filt_p


def load_model(model_name, num_classes, device):
    cfg = MODEL_CONFIGS[model_name]
    if model_name == 'homology':
        from train_homology_multimodal import HomologyMultimodalClassifier, HOMOLOGY_DIM
        model = HomologyMultimodalClassifier(num_classes=num_classes, esm_dim=1280,
                    gnn_hidden_dim=256, homology_dim=HOMOLOGY_DIM,
                    fusion_hidden_dim=512, dropout_rate=0.1)
    elif model_name == 'baseline':
        from multimodal_gnn_fixed import MultimodalToxinClassifier
        model = MultimodalToxinClassifier(esm_dim=1280, gnn_hidden_dim=256,
                    num_classes=num_classes, fusion_hidden_dim=512, dropout_rate=0.1)
    elif model_name == 'riemannian':
        from train_riemannian_multimodal import RiemannianToxinClassifier
        model = RiemannianToxinClassifier(num_classes=num_classes, esm_dim=1280,
                    gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1)
    elif model_name == 'invariant':
        from train_invariant_multimodal import InvariantMultimodalClassifier
        model = InvariantMultimodalClassifier(num_classes=num_classes, esm_dim=1280,
                    gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1)
    elif model_name == 'manifold':
        from train_statistical_manifold_multimodal import StatisticalManifoldClassifier
        model = StatisticalManifoldClassifier(num_classes=num_classes, esm_dim=1280,
                    gnn_hidden_dim=256, fusion_hidden_dim=512, dropout_rate=0.1)
    model.load_state_dict(torch.load(cfg['ckpt'], map_location=device))
    return model.to(device)

# ── Embedding extraction via forward hook ─────────────────────────────────────

def extract_embeddings(model, loader, device, model_name):
    """Hook onto classifier ReLU (index 1) to capture 512D fused representation."""
    model.eval()
    hook_outputs = []

    def hook_fn(module, input, output):
        hook_outputs.append(output.detach().cpu())

    handle = model.classifier[1].register_forward_hook(hook_fn)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if model_name == 'homology':
                g, esm, hom, y = batch
                g, esm, hom, y = g.to(device), esm.to(device), hom.to(device), y.to(device)
                logits, *_ = model(g, esm, hom)
            else:
                g, esm, y = batch
                g, esm, y = g.to(device), esm.to(device), y.to(device)
                logits, *_ = model(g, esm)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    handle.remove()

    embeddings = torch.cat(hook_outputs, dim=0).numpy()
    logits_np  = torch.cat(all_logits,   dim=0).numpy()
    labels_np  = torch.cat(all_labels,   dim=0).numpy()
    preds_np   = np.argmax(logits_np, axis=1)
    confidence = torch.softmax(torch.tensor(logits_np), dim=1).numpy().max(axis=1)
    return embeddings, preds_np, labels_np, confidence

# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(embeddings, pca_components=50, perplexity=40, random_state=42):
    print(f"  PCA: {embeddings.shape[1]}D → {pca_components}D ...")
    t0 = time.time()
    pca     = PCA(n_components=min(pca_components, embeddings.shape[1]), random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    print(f"  PCA done ({time.time()-t0:.1f}s), var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    print(f"  t-SNE: {reduced.shape[1]}D → 2D (perplexity={perplexity}) ...")
    t0 = time.time()
    coords = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                  random_state=random_state, verbose=1).fit_transform(reduced)
    print(f"  t-SNE done ({time.time()-t0:.1f}s)")
    return coords

# ── Static matplotlib ─────────────────────────────────────────────────────────

def plot_static(coords, labels, preds, taxa_labels, label_encoder, confidence, model_name):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        class_names  = label_encoder.classes_
        class_colors = {'Calcium': '#e74c3c', 'Potassium': '#2ecc71', 'Sodium': '#3498db'}
        taxa_colors  = {'Snake': '#e67e22', 'Cone Snail': '#9b59b6', 'Spider': '#1abc9c',
                        'Scorpion': '#f39c12', 'Sea Anemone': '#e91e63'}

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        title = f't-SNE of {model_name.title()} GNN Fusion Embeddings (512D → 2D)'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax = axes[0]
        for cls in class_names:
            mask = labels == label_encoder.transform([cls])[0]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=class_colors[cls],
                       label=cls, alpha=0.7, s=18, edgecolors='none')
        ax.set_title('By Ion Channel Class', fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.grid(alpha=0.2)

        ax = axes[1]
        for taxa in TAXA_LIST:
            mask = np.array(taxa_labels) == taxa
            ax.scatter(coords[mask, 0], coords[mask, 1], c=taxa_colors.get(taxa, '#999'),
                       label=taxa, alpha=0.7, s=18, edgecolors='none')
        ax.set_title('By Taxa', fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.grid(alpha=0.2)

        ax = axes[2]
        correct = preds == labels
        ax.scatter(coords[correct, 0],  coords[correct, 1],  c='#2ecc71',
                   label=f'Correct ({correct.sum()})',   alpha=0.6, s=18, edgecolors='none')
        ax.scatter(coords[~correct, 0], coords[~correct, 1], c='#e74c3c',
                   label=f'Wrong ({(~correct).sum()})', alpha=0.8, s=28,
                   edgecolors='white', linewidths=0.3)
        ax.set_title('Correct vs Wrong', fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.grid(alpha=0.2)

        plt.tight_layout()
        out = f'tsne_{model_name}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()
    except Exception as e:
        print(f"Static plot skipped: {e}")

# ── Interactive Plotly ────────────────────────────────────────────────────────

def plot_interactive(coords, labels, preds, taxa_labels, protein_ids,
                     label_encoder, confidence, model_name):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        class_names  = label_encoder.classes_
        class_colors = {'Calcium': '#e74c3c', 'Potassium': '#2ecc71', 'Sodium': '#3498db'}
        taxa_colors  = {'Snake': '#e67e22', 'Cone Snail': '#9b59b6', 'Spider': '#1abc9c',
                        'Scorpion': '#f39c12', 'Sea Anemone': '#e91e63'}

        labels_str  = np.array([class_names[i] for i in labels])
        preds_str   = np.array([class_names[i] for i in preds])
        correct_str = np.array(['✓ Correct' if p == l else '✗ Wrong'
                                for p, l in zip(preds, labels)])
        hover_text  = [
            f"<b>{pid}</b><br>Taxa: {taxa}<br>True: {true}<br>"
            f"Pred: {pred}<br>Confidence: {conf*100:.1f}%<br>{corr}"
            for pid, taxa, true, pred, conf, corr
            in zip(protein_ids, taxa_labels, labels_str, preds_str, confidence, correct_str)
        ]

        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
            subplot_titles=['By Ion Channel Class', 'By Taxa', 'Correct vs Wrong'])

        for cls in class_names:
            mask = labels_str == cls
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode='markers', name=cls,
                text=[hover_text[i] for i in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(color=class_colors[cls], size=6, opacity=0.75),
                legendgroup=cls,
            ), row=1, col=1)

        for taxa in TAXA_LIST:
            mask = np.array(taxa_labels) == taxa
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode='markers', name=taxa,
                text=[hover_text[i] for i in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(color=taxa_colors.get(taxa, '#999'), size=6, opacity=0.75),
                legendgroup=taxa,
            ), row=1, col=2)

        for label, color, sz in [('✓ Correct', '#2ecc71', 6), ('✗ Wrong', '#e74c3c', 9)]:
            mask = correct_str == label
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode='markers',
                name=f'{label} ({mask.sum()})',
                text=[hover_text[i] for i in np.where(mask)[0]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(color=color, size=sz, opacity=0.75),
                legendgroup=label,
            ), row=1, col=3)

        fig.update_layout(
            title=f't-SNE — {model_name.title()} GNN (512D → 2D) — hover for protein details',
            height=550, width=1500, template='plotly_white', hovermode='closest',
            legend=dict(itemsizing='constant', font=dict(size=11)),
        )
        fig.update_xaxes(title_text='t-SNE 1', showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(title_text='t-SNE 2', showgrid=True, gridcolor='rgba(0,0,0,0.1)')

        out = f'tsne_{model_name}_interactive.html'
        fig.write_html(out, include_plotlyjs='cdn')
        print(f"Saved: {out}")
    except Exception as e:
        print(f"Interactive plot skipped: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['homology', 'baseline', 'riemannian', 'invariant', 'manifold'],
                        help='Which model to visualize')
    parser.add_argument('--perplexity', type=int, default=40, help='t-SNE perplexity')
    parser.add_argument('--recompute', action='store_true', help='Ignore cached embeddings')
    args = parser.parse_args()

    model_name   = args.model
    embed_cache  = f'embeddings_{model_name}.pkl'
    tsne_cache   = f'tsne_data_{model_name}.pkl'

    print(f"=== XAI Embedding Visualization — {model_name.title()} GNN ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if Path(embed_cache).exists() and not args.recompute:
        print(f"\nLoading cached embeddings: {embed_cache}")
        with open(embed_cache, 'rb') as f:
            cache = pickle.load(f)
        embeddings = cache['embeddings'];  preds      = cache['preds']
        labels_out = cache['labels'];      confidence = cache['confidence']
        taxa_out   = cache['taxa_labels']; ids_out    = cache['protein_ids']
        label_enc  = cache['label_encoder']
    else:
        esm_features, metadata_df = load_base_data()
        labels, label_encoder, valid_proteins, taxa_labels, protein_ids = load_all_metadata(metadata_df)
        num_classes = len(label_encoder.classes_)

        print(f"\n=== Building Dataset for {model_name} ===")
        if model_name == 'homology':
            loader, filt_l, filt_p = build_dataset_homology(valid_proteins, esm_features, labels)
            filt_taxa = [taxa_labels[i] for i, p in enumerate(valid_proteins)
                         if any(fp['protein_id'] == p['protein_id'] for fp in filt_p)]
            filt_ids  = [protein_ids[i] for i, p in enumerate(valid_proteins)
                         if any(fp['protein_id'] == p['protein_id'] for fp in filt_p)]
        else:
            loader, filt_l, filt_p = build_dataset_standard(valid_proteins, esm_features, labels, model_name)
            p_set     = {p['protein_id'] for p in filt_p}
            filt_taxa = [t for p, t in zip(valid_proteins, taxa_labels) if p['protein_id'] in p_set]
            filt_ids  = [pid for p, pid in zip(valid_proteins, protein_ids) if p['protein_id'] in p_set]

        print(f"\n=== Loading Model ({MODEL_CONFIGS[model_name]['ckpt']}) ===")
        model = load_model(model_name, num_classes, device)

        print("\n=== Extracting 512D Fusion Embeddings ===")
        embeddings, preds, labels_out, confidence = extract_embeddings(model, loader, device, model_name)
        taxa_out  = filt_taxa
        ids_out   = filt_ids
        label_enc = label_encoder

        cache = {'embeddings': embeddings, 'preds': preds, 'labels': labels_out,
                 'confidence': confidence, 'taxa_labels': taxa_out,
                 'protein_ids': ids_out,   'label_encoder': label_enc}
        with open(embed_cache, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Embeddings cached: {embed_cache}")

    acc = (preds == labels_out).mean()
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Accuracy on full set: {acc*100:.1f}%")

    print("\n=== Running PCA → t-SNE ===")
    coords = run_tsne(embeddings, perplexity=args.perplexity)

    with open(tsne_cache, 'wb') as f:
        pickle.dump({'coords': coords, 'labels': labels_out, 'preds': preds,
                     'taxa_labels': taxa_out, 'protein_ids': ids_out,
                     'label_encoder': label_enc, 'confidence': confidence}, f)
    print(f"t-SNE coords saved: {tsne_cache}")

    print("\n=== Generating Plots ===")
    plot_static(coords, labels_out, preds, taxa_out, label_enc, confidence, model_name)
    plot_interactive(coords, labels_out, preds, taxa_out, ids_out, label_enc, confidence, model_name)

    print(f"\n=== Done ({model_name}) ===")
    print(f"  tsne_{model_name}.png")
    print(f"  tsne_{model_name}_interactive.html")
    print(f"  embeddings_{model_name}.pkl")
    print(f"  tsne_data_{model_name}.pkl")


if __name__ == "__main__":
    main()
