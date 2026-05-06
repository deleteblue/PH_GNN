#!/usr/bin/env python3
"""
Taxa-Based OOD Cross-Validation for Multimodal GNN.
Leave-one-taxa-out: train on 4 taxa, test on held-out taxa.
5 folds: Snake, Cone Snail, Spider, Scorpion, Sea Anemone
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pickle
import time
from pathlib import Path

from multimodal_gnn_fixed import MultimodalToxinClassifier

TAXA_LIST = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']

LABEL_NORMALIZE = {
    'Ca': 'Calcium',
    'Na': 'Sodium',
    'K':  'Potassium',
}

# ── Data loading (identical to train_full_multimodal.py) ──────────────────────

def load_fasta_based_data():
    print("=== Loading FASTA-Based Dataset ===")
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
    print(f"Loaded {len(esm_features)} ESM features, {len(metadata_df)} metadata entries")
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
        print(f"CONFLICT: {row['Entry']} - match='{row['match']}' vs Zhang='{row['Zhang result']}' - EXCLUDED")
        return None


def create_labels_and_taxa(metadata_df):
    """Return labels, valid_proteins list, and taxa per protein."""
    print("=== Creating Labels + Taxa Assignments ===")

    cone_df = pd.read_excel('cone_trainingData.xlsx')
    toxin_df  = pd.read_excel('toxinCombined_Finalv3.xlsx')
    taxa_df   = pd.read_csv('taxa_fastas/taxa_metadata.csv')[['protein_id', 'taxa']]
    taxa_lookup = dict(zip(taxa_df['protein_id'], taxa_df['taxa']))

    labels         = []
    valid_proteins = []
    taxa_labels    = []

    for _, row in metadata_df.iterrows():
        protein_id = row['protein_id']
        dataset    = row['dataset']

        if dataset == 'cone':
            protein_row = cone_df[cone_df['Entry'] == protein_id]
            if len(protein_row) == 0:
                continue
            target_class = determine_ion_channel_class(protein_row.iloc[0])
            if target_class is None:
                continue
            target_class = LABEL_NORMALIZE.get(target_class, target_class)

        elif dataset == 'toxin_combined':
            protein_row = toxin_df[toxin_df['entry'] == protein_id]
            if len(protein_row) == 0:
                continue
            target_class = protein_row.iloc[0]['ionChannelClass']

        else:
            continue

        taxa = taxa_lookup.get(protein_id)
        if taxa is None:
            continue

        labels.append(target_class)
        valid_proteins.append(row)
        taxa_labels.append(taxa)

    print(f"Found {len(labels)} labelled proteins")

    # Fit label encoder on ALL data so class indices are consistent across folds
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    print(f"Classes: {label_encoder.classes_}")
    for cls, cnt in zip(*np.unique(labels, return_counts=True)):
        print(f"  {cls}: {cnt}")

    print("\nTaxa distribution:")
    for taxa in TAXA_LIST:
        n = taxa_labels.count(taxa)
        print(f"  {taxa}: {n}")

    return encoded_labels, label_encoder, valid_proteins, taxa_labels

# ── Graph construction (identical to train_full_multimodal.py) ────────────────

def parse_pdb_to_tensors(pdb_path):
    atoms_data = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name    = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_num  = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip() or atom_name[0]
                atoms_data.append({
                    'atom_name': atom_name, 'residue_name': residue_name,
                    'residue_num': residue_num, 'x': x, 'y': y, 'z': z,
                    'element': element
                })

    coordinates   = torch.tensor([[a['x'], a['y'], a['z']] for a in atoms_data], dtype=torch.float)
    unique_elements = ['C', 'N', 'O', 'S', 'H']
    elem_to_idx   = {e: i for i, e in enumerate(unique_elements)}

    node_features = []
    for atom in atoms_data:
        one_hot = [0] * 5
        one_hot[elem_to_idx.get(atom['element'], 0)] = 1
        node_features.append(one_hot)

    return torch.tensor(node_features, dtype=torch.float), coordinates, atoms_data


def detect_disulfide_bonds(coordinates, atoms_data, threshold=2.5):
    cys_s = [i for i, a in enumerate(atoms_data)
             if a['residue_name'] == 'CYS' and a['atom_name'] == 'SG']
    bonds = []
    for i in range(len(cys_s)):
        for j in range(i + 1, len(cys_s)):
            if torch.norm(coordinates[cys_s[i]] - coordinates[cys_s[j]]).item() <= threshold:
                bonds.append((cys_s[i], cys_s[j]))
    return bonds


def build_spatial_edges(coordinates, disulfide_pairs, threshold=4.0):
    n = coordinates.shape[0]
    disulfide_set = set(disulfide_pairs) | {(j, i) for i, j in disulfide_pairs}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in disulfide_set:
                if torch.norm(coordinates[i] - coordinates[j]).item() <= threshold:
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
        x, pos, atoms = parse_pdb_to_tensors(protein_info['structure_file'])
        disulfide     = detect_disulfide_bonds(pos, atoms)
        return Data(
            x=x, pos=pos,
            edge_index_spatial=build_spatial_edges(pos, disulfide),
            edge_index_disulfide=build_disulfide_edges(disulfide),
        )
    except Exception as e:
        print(f"  Error {protein_info['protein_id']}: {e}")
        return None


def build_all_graphs(valid_proteins):
    print("=== Building Protein Graphs ===")
    graphs = []
    failed = 0
    for i, p in enumerate(valid_proteins):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(valid_proteins)}")
        g = create_protein_graph(p)
        if g is not None:
            graphs.append(g)
        else:
            failed += 1
    print(f"  Built {len(graphs)} graphs ({failed} failed)")
    return graphs

# ── Training / evaluation ─────────────────────────────────────────────────────

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, esm, labels):
        self.graphs = graphs
        self.esm    = esm
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.esm[idx], self.labels[idx]


def collate_fn(batch):
    graphs, esm, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(labels)


def make_loader(graphs, esm, labels, batch_size=16, shuffle=False):
    ds = MultimodalDataset(graphs, esm, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for graph_batch, esm_batch, labels_batch in loader:
        graph_batch  = graph_batch.to(device)
        esm_batch    = esm_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(graph_batch, esm_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == labels_batch).sum().item()
        n       += labels_batch.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), correct / n


def evaluate(model, loader, criterion, device, label_encoder, verbose=False):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for graph_batch, esm_batch, labels_batch in loader:
            graph_batch  = graph_batch.to(device)
            esm_batch    = esm_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits, _, _ = model(graph_batch, esm_batch)
            total_loss  += criterion(logits, labels_batch).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(labels_batch.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc      = (np.array(preds) == np.array(targets)).mean()
    if verbose:
        print(classification_report(targets, preds, target_names=label_encoder.classes_))
    return avg_loss, acc, preds, targets


def train_fold(train_graphs, train_esm, train_labels,
               test_graphs,  test_esm,  test_labels,
               label_encoder, num_classes, device, num_epochs=50,
               patience=10, fold_name='fold'):
    """Train one fold from scratch with early stopping, return loss/acc curves + final eval."""

    train_loader = make_loader(train_graphs, train_esm, train_labels, shuffle=True)
    test_loader  = make_loader(test_graphs,  test_esm,  test_labels,  shuffle=False)

    # Class-weighted loss — compensates for Calcium being outnumbered ~3:1
    class_counts  = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = MultimodalToxinClassifier(
        esm_dim=1280, gnn_hidden_dim=256,
        num_classes=num_classes, fusion_hidden_dim=512, dropout_rate=0.15
    ).to(device)
    # lr 0.001→0.0003 (slower memorization), weight_decay=1e-4 (L2), dropout 0.1→0.15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0
    ckpt_path        = f'ood_best_{fold_name.replace(" ", "_")}.pth'

    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, _, _ = evaluate(model, test_loader, criterion, device, label_encoder)

        train_losses.append(t_loss)
        train_accs.append(t_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>2}/{num_epochs} | "
                  f"Train {t_loss:.4f}/{t_acc*100:.1f}%  "
                  f"Val {v_loss:.4f}/{v_acc*100:.1f}%")

        if v_loss < best_val_loss - 1e-4:
            best_val_loss    = v_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    # Load best checkpoint before final evaluation
    model.load_state_dict(torch.load(ckpt_path))
    print(f"  Loaded best model from epoch {best_epoch}")
    print()

    _, final_acc, final_preds, final_targets = evaluate(
        model, test_loader, criterion, device, label_encoder, verbose=True
    )

    return {
        'train_losses': train_losses, 'train_accs': train_accs,
        'val_losses':   val_losses,   'val_accs':   val_accs,
        'final_acc': final_acc, 'preds': final_preds, 'targets': final_targets,
        'best_epoch': best_epoch,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Taxa OOD Cross-Validation — Multimodal GNN ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load everything once
    esm_features, metadata_df = load_fasta_based_data()
    labels, label_encoder, valid_proteins, taxa_labels = create_labels_and_taxa(metadata_df)
    num_classes = len(label_encoder.classes_)

    print("\n=== Building All Protein Graphs (done once) ===")
    graphs = build_all_graphs(valid_proteins)

    # Remove any failed graphs
    ok = [i for i, g in enumerate(graphs) if g is not None]
    graphs       = [graphs[i]       for i in ok]
    taxa_labels  = [taxa_labels[i]  for i in ok]
    labels       = [labels[i]       for i in ok]
    valid_proteins = [valid_proteins[i] for i in ok]

    # Stack ESM features
    esm_tensors = torch.stack([
        torch.from_numpy(esm_features[p['feature_key']]).float()
        for p in valid_proteins
    ])
    labels_tensor = torch.from_numpy(np.array(labels)).long()

    taxa_array = np.array(taxa_labels)
    print(f"\nReady: {len(graphs)} proteins across {len(TAXA_LIST)} taxa\n")

    # ── OOD loop ──
    all_results  = {}
    total_start  = time.time()

    for held_out in TAXA_LIST:
        print(f"\n{'='*60}")
        print(f"  HELD-OUT TAXA: {held_out}")
        print(f"{'='*60}")

        test_mask  = (taxa_array == held_out)
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        print(f"  Train: {len(train_idx)} proteins | Test: {len(test_idx)} proteins")

        # Class distribution in test set
        test_classes, test_counts = np.unique(labels_tensor[test_idx].numpy(), return_counts=True)
        print("  Test class distribution: " +
              ", ".join(f"{label_encoder.classes_[c]}: {n}" for c, n in zip(test_classes, test_counts)))

        fold_start = time.time()
        result = train_fold(
            train_graphs=[graphs[i]       for i in train_idx],
            train_esm=esm_tensors[train_idx],
            train_labels=labels_tensor[train_idx],
            test_graphs=[graphs[i]        for i in test_idx],
            test_esm=esm_tensors[test_idx],
            test_labels=labels_tensor[test_idx],
            label_encoder=label_encoder,
            num_classes=num_classes,
            device=device,
            fold_name=held_out,
        )
        result['fold_time'] = time.time() - fold_start
        all_results[held_out] = result
        print(f"  Fold time: {result['fold_time']/60:.1f} min | Final OOD acc: {result['final_acc']*100:.1f}%")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Total training time: {total_time/3600:.2f} hours")
    print(f"{'='*60}")

    # ── Summary table ──
    print("\n=== OOD Results Summary ===")
    print(f"{'Held-out Taxa':<15} {'Test Acc':>10} {'Best Val Acc':>14} {'Time (min)':>12}")
    print("-" * 55)
    accs = []
    for taxa in TAXA_LIST:
        r   = all_results[taxa]
        acc = r['final_acc'] * 100
        best_val = max(r['val_accs']) * 100
        accs.append(acc)
        print(f"{taxa:<15} {acc:>9.1f}%  {best_val:>12.1f}%  {r['fold_time']/60:>10.1f}")
    print("-" * 55)
    print(f"{'Mean':<15} {np.mean(accs):>9.1f}%")
    print(f"{'Std':<15} {np.std(accs):>9.1f}%")

    # ── Save results ──
    with open('ood_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("\nResults saved: ood_results.pkl")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = range(1, 51)

        # Figure 1: loss curves per fold (1 row × 5 cols)
        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
        fig.suptitle('Multimodal GNN — Taxa OOD: Train vs Val Loss per Fold', fontsize=13, fontweight='bold')
        for ax, taxa in zip(axes, TAXA_LIST):
            r = all_results[taxa]
            ax.plot(epochs, r['train_losses'], color='steelblue', label='Train')
            ax.plot(epochs, r['val_losses'],   color='tomato', linestyle='--', label='Val (OOD)')
            ax.set_title(f'Test: {taxa}', fontsize=10)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('ood_loss_curves.png', dpi=150, bbox_inches='tight')
        print("Plot saved: ood_loss_curves.png")

        # Figure 2: OOD test accuracy bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple', 'tomato']
        bars = ax2.bar(TAXA_LIST, [all_results[t]['final_acc'] * 100 for t in TAXA_LIST],
                       color=colors, edgecolor='white', linewidth=1.2)
        ax2.axhline(np.mean(accs), color='black', linestyle='--', linewidth=1.5,
                    label=f'Mean: {np.mean(accs):.1f}%')
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel('OOD Test Accuracy (%)')
        ax2.set_title('Multimodal GNN — Taxa OOD Accuracy per Held-Out Taxa', fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('ood_accuracy_summary.png', dpi=150, bbox_inches='tight')
        print("Plot saved: ood_accuracy_summary.png")

    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
