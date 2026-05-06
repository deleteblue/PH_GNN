#!/usr/bin/env python3
"""
XAI data extraction for HomologyMultimodalClassifier.
Run once via SLURM to precompute xai_data.pt, then use xai_dash_app.py locally.

Extracts per-protein:
  - GAT attention weights (per atom, aggregated to residues)
  - Fused embedding (struct + seq + homology, 768D) → UMAP
  - Predicted label + confidence per class
  - True label

Output: xai_data.pt
"""

import sys, os, importlib_metadata
sys.modules['importlib.metadata'] = importlib_metadata

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from multimodal_gnn_fixed import StructuralEncoder, SequenceEncoder
from train_homology_multimodal import (
    HomologyMultimodalClassifier,
    load_fasta_based_data,
    create_labels_from_metadata,
    create_protein_graphs_batch,
    compute_homology_features_batch,
    HOMOLOGY_DIM,
)
from train_full_multimodal import parse_pdb_to_tensors

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_WEIGHTS = 'homology_multimodal_gnn.pth'
OUTPUT_FILE   = 'xai_data.pt'

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading ESM features + metadata...")
esm_features, metadata_df = load_fasta_based_data()
labels, label_encoder, valid_proteins = create_labels_from_metadata(metadata_df)

print("Building protein graphs...")
graphs, proteins = create_protein_graphs_batch(valid_proteins)

# Rebuild filtered labels to match the proteins that had valid graphs
protein_id_to_label = {p['protein_id']: labels[i] for i, p in enumerate(valid_proteins)}
filt_labels = [protein_id_to_label[p['protein_id']] for p in proteins]

esm_tensors = torch.stack([
    torch.from_numpy(esm_features[p['feature_key']]).float()
    for p in proteins
])
print(f"Proteins with structures: {len(graphs)}")

print("Computing homology features...")
homology_tensors = compute_homology_features_batch(proteins)

# ── Load homology model ───────────────────────────────────────────────────────
device = torch.device('cpu')
model = HomologyMultimodalClassifier(
    num_classes=len(label_encoder.classes_),
    esm_dim=1280,
    gnn_hidden_dim=256,
    homology_dim=HOMOLOGY_DIM,
    fusion_hidden_dim=512,
    dropout_rate=0.1,
).to(device)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, MODEL_WEIGHTS), map_location=device
))
model.eval()
print(f"Loaded: {MODEL_WEIGHTS}  ({sum(p.numel() for p in model.parameters()):,} params)")

# ── Attention + embedding extraction (one protein at a time) ──────────────────
def extract_one(graph, esm_vec, hom_vec):
    """
    Single-protein forward pass.
    Returns:
        node_attn : np.array [num_atoms]  — per-atom attention (aggregated from all GAT layers)
        embedding : np.array [768]        — struct+seq+homology fused representation
    """
    from torch_geometric.data import Batch as PyGBatch

    batch  = PyGBatch.from_data_list([graph])
    esm_b  = esm_vec.unsqueeze(0)
    hom_b  = hom_vec.unsqueeze(0)
    enc    = model.structural_encoder

    with torch.no_grad():
        # ── Structural encoder with attention capture ──────────────────────────
        node_input = torch.cat([batch.x, batch.pos], dim=1)
        h = F.relu(enc.node_embedding(node_input))

        spatial_h  = h
        all_alpha, all_ei = [], []
        for conv in enc.spatial_convs:
            out, (ei, alpha) = conv(spatial_h, batch.edge_index_spatial,
                                    return_attention_weights=True)
            spatial_h = F.relu(out)
            all_alpha.append(alpha.detach())  # [E, heads]
            all_ei.append(ei)

        if batch.edge_index_disulfide.shape[1] > 0:
            disulfide_h = F.relu(enc.disulfide_attention(h, batch.edge_index_disulfide))
        else:
            disulfide_h = torch.zeros_like(spatial_h)

        combined  = torch.cat([spatial_h, disulfide_h], dim=1)
        fused_h   = F.relu(enc.fusion(combined))
        struct_out = fused_h.mean(dim=0, keepdim=True)         # [1, 256]

        # ── Sequence + homology encoders ───────────────────────────────────────
        seq_out = model.sequence_encoder(esm_b)                # [1, 256]
        hom_out = model.homology_encoder(hom_b)                # [1, 256]

        # Fused embedding (before classifier) — 768D
        embedding = torch.cat([struct_out, seq_out, hom_out], dim=1).squeeze(0)  # [768]

        # ── Per-node attention: mean over all layers and heads ─────────────────
        num_nodes = batch.num_nodes
        node_attn_sum = torch.zeros(num_nodes)
        node_degree   = torch.zeros(num_nodes)

        for ei, alpha in zip(all_ei, all_alpha):
            alpha_mean = alpha.mean(dim=-1)   # mean over heads → [E]
            tgt = ei[1]                        # target node per edge
            for t, a in zip(tgt.tolist(), alpha_mean.tolist()):
                node_attn_sum[t] += a
                node_degree[t]   += 1

        node_degree   = node_degree.clamp(min=1)
        node_attn     = (node_attn_sum / node_degree).numpy()

    return node_attn, embedding.numpy()


print("Extracting attention + embeddings...")
t0 = time.time()

embeddings         = []
node_attentions    = []
residue_attentions = []
pdb_paths_out      = []

for i, (graph, pinfo, esm, hom) in enumerate(
        zip(graphs, proteins, esm_tensors, homology_tensors)):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(graphs)}  ({time.time()-t0:.0f}s)")

    node_attn, emb = extract_one(graph, esm, hom)
    embeddings.append(emb)
    node_attentions.append(node_attn)

    pdb_path = pinfo['structure_file']
    pdb_paths_out.append(pdb_path)

    # Map atoms → residues (mean attention per residue)
    try:
        _, _, atoms_data = parse_pdb_to_tensors(pdb_path)
        resid_sum, resid_count = {}, {}
        for atom_idx, atom in enumerate(atoms_data):
            if atom_idx >= len(node_attn):
                break
            key = (atom['residue_num'], atom.get('residue_name', 'UNK'))
            resid_sum[key]   = resid_sum.get(key, 0.0) + float(node_attn[atom_idx])
            resid_count[key] = resid_count.get(key, 0)  + 1
        res_attn = {k: resid_sum[k] / resid_count[k] for k in resid_sum}
    except Exception:
        res_attn = {}

    residue_attentions.append(res_attn)

print(f"Extraction done in {time.time()-t0:.1f}s")

# ── Predictions + confidence scores ──────────────────────────────────────────
print("Running predictions...")
from torch_geometric.data import Batch as PyGBatch

predictions, confidences = [], []
BATCH_SIZE = 32

for start in range(0, len(graphs), BATCH_SIZE):
    end   = min(start + BATCH_SIZE, len(graphs))
    g_b   = PyGBatch.from_data_list(graphs[start:end])
    esm_b = esm_tensors[start:end]
    hom_b = homology_tensors[start:end]
    with torch.no_grad():
        logits, *_ = model(g_b, esm_b, hom_b)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    predictions.extend(probs.argmax(axis=-1).tolist())
    confidences.extend(probs.tolist())

predictions = np.array(predictions)
confidences = np.array(confidences)
correct     = (predictions == np.array(filt_labels)).astype(int)
print(f"Overall accuracy: {correct.mean()*100:.1f}%  ({correct.sum()}/{len(correct)} correct)")

# ── UMAP ──────────────────────────────────────────────────────────────────────
print("Running UMAP on 768D fused embeddings...")
import umap as umap_lib
embeddings_matrix = np.stack(embeddings)
reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_coords = reducer.fit_transform(embeddings_matrix)
print(f"UMAP done: {umap_coords.shape}")

# ── Taxa lookup ───────────────────────────────────────────────────────────────
taxa_df     = pd.read_csv(os.path.join(BASE_DIR, 'taxa_fastas/taxa_metadata.csv'))
taxa_lookup = dict(zip(taxa_df['protein_id'], taxa_df['taxa']))
taxa_list   = [taxa_lookup.get(p['protein_id'], 'Unknown') for p in proteins]

# ── Save ──────────────────────────────────────────────────────────────────────
xai_data = {
    'umap_coords':        umap_coords,
    'embeddings':         embeddings_matrix,
    'labels':             np.array(filt_labels),
    'label_names':        label_encoder.classes_.tolist(),
    'predictions':        predictions,
    'confidences':        confidences,
    'correct':            correct,
    'protein_ids':        [p['protein_id'] for p in proteins],
    'pdb_paths':          pdb_paths_out,
    'taxa':               taxa_list,
    'node_attentions':    node_attentions,
    'residue_attentions': residue_attentions,
    'model_used':         MODEL_WEIGHTS,
}

torch.save(xai_data, os.path.join(BASE_DIR, OUTPUT_FILE))
print(f"Saved {OUTPUT_FILE}  ({len(graphs)} proteins)")
