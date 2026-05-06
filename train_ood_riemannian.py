#!/usr/bin/env python3
"""
Taxa-Based OOD Cross-Validation — Riemannian Multimodal GNN.
Same leave-one-taxa-out protocol as train_ood_multimodal.py but uses
RiemannianToxinClassifier (learnable metric factor on ESM space).
Includes: early stopping (patience=10) + class-weighted loss per fold.
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

from train_riemannian_multimodal import RiemannianToxinClassifier

TAXA_LIST = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']

LABEL_NORMALIZE = {'Ca': 'Calcium', 'Na': 'Sodium', 'K': 'Potassium'}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_fasta_based_data():
    print("=== Loading FASTA-Based Dataset ===")
    esm_features = torch.load('esm_features_fasta_aligned.pt')
    metadata_df  = pd.read_csv('feature_metadata_fasta_aligned.csv')
    print(f"Loaded {len(esm_features)} ESM features")
    return esm_features, metadata_df


def determine_ion_channel_class(row):
    if row['match'] == row['Zhang result']:
        return row['match']
    elif row['match'] == 'Other':
        return row['Zhang result']
    elif row['Zhang result'] == 'Other':
        return row['match']
    else:
        print(f"CONFLICT: {row['Entry']} - excluded")
        return None


def create_labels_and_taxa(metadata_df):
    print("=== Creating Labels + Taxa Assignments ===")
    cone_df   = pd.read_excel('cone_trainingData.xlsx')
    toxin_df   = pd.read_excel('toxinCombined_Finalv3.xlsx')
    taxa_df    = pd.read_csv('taxa_fastas/taxa_metadata.csv')[['protein_id', 'taxa']]
    taxa_lookup = dict(zip(taxa_df['protein_id'], taxa_df['taxa']))

    labels, valid_proteins, taxa_labels = [], [], []
    for _, row in metadata_df.iterrows():
        pid, dataset = row['protein_id'], row['dataset']
        if dataset == 'cone':
            protein_row = cone_df[cone_df['Entry'] == pid]
            if len(protein_row) == 0: continue
            cls = determine_ion_channel_class(protein_row.iloc[0])
            if cls is None: continue
            cls = LABEL_NORMALIZE.get(cls, cls)
        elif dataset == 'toxin_combined':
            protein_row = toxin_df[toxin_df['entry'] == pid]
            if len(protein_row) == 0: continue
            cls = protein_row.iloc[0]['ionChannelClass']
        else:
            continue
        taxa = taxa_lookup.get(pid)
        if taxa is None: continue
        labels.append(cls)
        valid_proteins.append(row)
        taxa_labels.append(taxa)

    print(f"Found {len(labels)} labelled proteins")
    label_encoder  = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    print(f"Classes: {label_encoder.classes_}")
    for cls, cnt in zip(*np.unique(labels, return_counts=True)):
        print(f"  {cls}: {cnt}")
    print("\nTaxa distribution:")
    for taxa in TAXA_LIST:
        print(f"  {taxa}: {taxa_labels.count(taxa)}")
    return encoded_labels, label_encoder, valid_proteins, taxa_labels

# ── Graph construction ────────────────────────────────────────────────────────

def parse_pdb_to_tensors(pdb_path):
    atoms_data = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atoms_data.append({
                    'atom_name': line[12:16].strip(), 'residue_name': line[17:20].strip(),
                    'x': float(line[30:38]), 'y': float(line[38:46]), 'z': float(line[46:54]),
                    'element': line[76:78].strip() or line[12:16].strip()[0]
                })
    coords = torch.tensor([[a['x'], a['y'], a['z']] for a in atoms_data], dtype=torch.float)
    elem_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'H': 4}
    node_feats = []
    for a in atoms_data:
        oh = [0]*5; oh[elem_map.get(a['element'], 0)] = 1
        node_feats.append(oh)
    return torch.tensor(node_feats, dtype=torch.float), coords, atoms_data


def detect_disulfide_bonds(coords, atoms_data, threshold=2.5):
    cys_sg = [i for i, a in enumerate(atoms_data) if a['residue_name']=='CYS' and a['atom_name']=='SG']
    bonds = []
    for i in range(len(cys_sg)):
        for j in range(i+1, len(cys_sg)):
            if torch.norm(coords[cys_sg[i]] - coords[cys_sg[j]]).item() <= threshold:
                bonds.append((cys_sg[i], cys_sg[j]))
    return bonds


def build_spatial_edges(coords, disulfide_pairs, threshold=4.0):
    n = coords.shape[0]
    dsulf_set = set(disulfide_pairs) | {(j,i) for i,j in disulfide_pairs}
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) not in dsulf_set and torch.norm(coords[i]-coords[j]).item() <= threshold:
                edges.extend([(i,j),(j,i)])
    if not edges: return torch.empty((2,0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t()


def build_disulfide_edges(bonds):
    if not bonds: return torch.empty((2,0), dtype=torch.long)
    edges = [(i,j) for i,j in bonds] + [(j,i) for i,j in bonds]
    return torch.tensor(edges, dtype=torch.long).t()


def create_protein_graph(protein_info):
    try:
        x, pos, atoms = parse_pdb_to_tensors(protein_info['structure_file'])
        dsulf = detect_disulfide_bonds(pos, atoms)
        return Data(x=x, pos=pos,
                    edge_index_spatial=build_spatial_edges(pos, dsulf),
                    edge_index_disulfide=build_disulfide_edges(dsulf))
    except Exception as e:
        print(f"  Error {protein_info['protein_id']}: {e}")
        return None


def build_all_graphs(valid_proteins):
    print("=== Building Protein Graphs ===")
    graphs, failed = [], 0
    for i, p in enumerate(valid_proteins):
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(valid_proteins)}")
        g = create_protein_graph(p)
        if g is not None: graphs.append(g)
        else: failed += 1
    print(f"  Built {len(graphs)} graphs ({failed} failed)")
    return graphs

# ── Training ──────────────────────────────────────────────────────────────────

class RiemannianDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, esm, labels):
        self.graphs = graphs; self.esm = esm; self.labels = labels
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.esm[idx], self.labels[idx]


def collate_fn(batch):
    graphs, esm, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(esm), torch.stack(labels)


def make_loader(graphs, esm, labels, batch_size=16, shuffle=False):
    return DataLoader(RiemannianDataset(graphs, esm, labels),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for g, esm, y in loader:
        g, esm, y = g.to(device), esm.to(device), y.to(device)
        optimizer.zero_grad()
        logits, *_ = model(g, esm)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0); total_loss += loss.item()
    return total_loss / len(loader), correct / n


def evaluate(model, loader, criterion, device, label_encoder, verbose=False):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for g, esm, y in loader:
            g, esm, y = g.to(device), esm.to(device), y.to(device)
            logits, *_ = model(g, esm)
            total_loss += criterion(logits, y).item()
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = (np.array(preds) == np.array(targets)).mean()
    if verbose:
        print(classification_report(targets, preds, target_names=label_encoder.classes_))
    return total_loss / len(loader), acc, preds, targets


def train_fold(train_graphs, train_esm, train_labels,
               test_graphs, test_esm, test_labels,
               label_encoder, num_classes, device,
               num_epochs=50, patience=10, fold_name='fold'):

    train_loader = make_loader(train_graphs, train_esm, train_labels, shuffle=True)
    test_loader  = make_loader(test_graphs,  test_esm,  test_labels,  shuffle=False)

    # Class-weighted loss per fold
    class_counts  = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = RiemannianToxinClassifier(
        esm_dim=1280, gnn_hidden_dim=256, num_classes=num_classes,
        metric_rank=64, fusion_hidden_dim=512, dropout_rate=0.1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_loss, patience_counter, best_epoch = float('inf'), 0, 0
    ckpt = f'ood_riemannian_best_{fold_name.replace(" ","_")}.pth'

    for epoch in range(num_epochs):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, _, _ = evaluate(model, test_loader, criterion, device, label_encoder)
        train_losses.append(t_loss); train_accs.append(t_acc)
        val_losses.append(v_loss);   val_accs.append(v_acc)

        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1:>2}/{num_epochs} | Train {t_loss:.4f}/{t_acc*100:.1f}%  Val {v_loss:.4f}/{v_acc*100:.1f}%")

        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss; best_epoch = epoch+1; patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(torch.load(ckpt))
    print(f"  Loaded best model from epoch {best_epoch}")

    # Log learned metric singular values
    sv = model.sequence_encoder.analyze_metric()
    print(f"  Metric top-5 singular values: {sv[:5].round(4)}")
    print()

    _, final_acc, final_preds, final_targets = evaluate(
        model, test_loader, criterion, device, label_encoder, verbose=True)

    return {
        'train_losses': train_losses, 'train_accs': train_accs,
        'val_losses': val_losses, 'val_accs': val_accs,
        'final_acc': final_acc, 'preds': final_preds, 'targets': final_targets,
        'best_epoch': best_epoch, 'metric_sv': sv[:10].tolist(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Taxa OOD Cross-Validation — Riemannian Multimodal GNN ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    esm_features, metadata_df = load_fasta_based_data()
    labels, label_encoder, valid_proteins, taxa_labels = create_labels_and_taxa(metadata_df)
    num_classes = len(label_encoder.classes_)

    print("\n=== Building All Protein Graphs (done once) ===")
    graphs = build_all_graphs(valid_proteins)

    ok = [i for i, g in enumerate(graphs) if g is not None]
    graphs         = [graphs[i]        for i in ok]
    taxa_labels    = [taxa_labels[i]   for i in ok]
    labels         = [labels[i]        for i in ok]
    valid_proteins = [valid_proteins[i] for i in ok]

    esm_tensors   = torch.stack([torch.from_numpy(esm_features[p['feature_key']]).float() for p in valid_proteins])
    labels_tensor = torch.from_numpy(np.array(labels)).long()
    taxa_array    = np.array(taxa_labels)
    print(f"\nReady: {len(graphs)} proteins across {len(TAXA_LIST)} taxa\n")

    all_results = {}
    total_start = time.time()

    for held_out in TAXA_LIST:
        print(f"\n{'='*60}")
        print(f"  HELD-OUT TAXA: {held_out}")
        print(f"{'='*60}")

        test_mask  = (taxa_array == held_out)
        train_mask = ~test_mask
        train_idx  = np.where(train_mask)[0]
        test_idx   = np.where(test_mask)[0]

        print(f"  Train: {len(train_idx)} proteins | Test: {len(test_idx)} proteins")
        test_classes, test_counts = np.unique(labels_tensor[test_idx].numpy(), return_counts=True)
        print("  Test class distribution: " +
              ", ".join(f"{label_encoder.classes_[c]}: {n}" for c, n in zip(test_classes, test_counts)))

        fold_start = time.time()
        result = train_fold(
            train_graphs=[graphs[i] for i in train_idx],
            train_esm=esm_tensors[train_idx],
            train_labels=labels_tensor[train_idx],
            test_graphs=[graphs[i] for i in test_idx],
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

    print("\n=== OOD Results Summary (Riemannian) ===")
    print(f"{'Held-out Taxa':<15} {'Test Acc':>10} {'Best Val':>10} {'Best Epoch':>12} {'Time (min)':>12}")
    print("-" * 62)
    accs = []
    for taxa in TAXA_LIST:
        r    = all_results[taxa]
        acc  = r['final_acc'] * 100
        best = max(r['val_accs']) * 100
        accs.append(acc)
        print(f"{taxa:<15} {acc:>9.1f}%  {best:>9.1f}%  {r['best_epoch']:>10}  {r['fold_time']/60:>10.1f}")
    print("-" * 62)
    print(f"{'Mean':<15} {np.mean(accs):>9.1f}%")
    print(f"{'Std':<15} {np.std(accs):>9.1f}%")

    with open('ood_riemannian_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("\nResults saved: ood_riemannian_results.pkl")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
        fig.suptitle('Riemannian OOD — Train vs Val Accuracy per Fold', fontsize=12, fontweight='bold')
        colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple', 'tomato']
        for ax, taxa, col in zip(axes, TAXA_LIST, colors):
            r  = all_results[taxa]
            ep = range(1, len(r['train_accs']) + 1)
            ax.plot(ep, [a*100 for a in r['train_accs']], color=col, label='Train')
            ax.plot(ep, [a*100 for a in r['val_accs']],   color=col, linestyle='--', alpha=0.7, label='Val')
            ax.axvline(r['best_epoch'], color='black', linestyle=':', linewidth=1, label=f'Best ep {r["best_epoch"]}')
            ax.set_title(f'Test: {taxa}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('ood_riemannian_curves.png', dpi=150, bbox_inches='tight')
        print("Plot saved: ood_riemannian_curves.png")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar(TAXA_LIST, accs, color=colors, edgecolor='white')
        ax2.axhline(np.mean(accs), color='black', linestyle='--', label=f'Mean: {np.mean(accs):.1f}%')
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
        ax2.set_ylabel('OOD Accuracy (%)'); ax2.set_ylim(0, 105)
        ax2.set_title('Riemannian OOD Accuracy per Held-Out Taxa', fontweight='bold')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('ood_riemannian_summary.png', dpi=150, bbox_inches='tight')
        print("Plot saved: ood_riemannian_summary.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
