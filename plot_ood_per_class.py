#!/usr/bin/env python3
"""
Per-class F1 comparison across all OOD models.
Reads directly from saved pkl files — no model loading needed.
Output: ood_per_class_f1.png
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

TAXA_LIST = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']
CLASSES   = ['Calcium', 'Potassium', 'Sodium']

MODELS = [
    ('Homology',        'ood_homology_results.pkl'),
    ('Stat. Manifold',  'ood_manifold_results.pkl'),
    ('Baseline',        'ood_results.pkl'),
    ('Invariant SE(3)', 'ood_invariant_results.pkl'),
    ('Riemannian',      'ood_riemannian_results.pkl'),
    ('Spectral',        'ood_spectral_results.pkl'),
]

CLASS_COLORS = ['#1f77b4', '#2ca02c', '#d62728']   # blue, green, red


def f1_per_class(preds, targets, n_classes=3):
    preds, targets = np.array(preds), np.array(targets)
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        # if class has no true samples, F1 is undefined → 0
        if (tp + fn) == 0:
            f1s.append(np.nan)
            continue
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return f1s


def load_results(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Build data ────────────────────────────────────────────────────────────────

available = []
for model_name, pkl_path in MODELS:
    d = load_results(pkl_path)
    if d is not None:
        available.append((model_name, d))
        print(f"Loaded: {model_name}  ({pkl_path})")
    else:
        print(f"Missing: {model_name}  ({pkl_path}) — skipped")

n_models = len(available)

# ── Figure 1: Per-model per-fold per-class F1 bar chart ──────────────────────
# Layout: n_models rows × 5 cols (one col per fold)

fig, axes = plt.subplots(n_models, 5, figsize=(20, 3.5 * n_models),
                         sharey=True, constrained_layout=True)
fig.suptitle('OOD Per-Class F1 by Model and Held-Out Taxa', fontsize=14, fontweight='bold')

x    = np.arange(3)
w    = 0.55
tick_labels = ['Ca', 'K', 'Na']

for row, (model_name, results) in enumerate(available):
    for col, taxa in enumerate(TAXA_LIST):
        ax = axes[row, col] if n_models > 1 else axes[col]
        r  = results.get(taxa, results.get(taxa.replace(' ', '_')))
        if r is None:
            ax.set_visible(False)
            continue

        preds   = r['preds']
        targets = r.get('targets', r.get('trues', []))
        f1s     = f1_per_class(preds, targets)

        bars = ax.bar(x, [0 if np.isnan(v) else v for v in f1s],
                      color=CLASS_COLORS, width=w, edgecolor='white', linewidth=0.5)

        # mark undefined (no true samples) with hatching
        for i, v in enumerate(f1s):
            if np.isnan(v):
                ax.bar(x[i], 0.05, color='lightgrey', width=w,
                       hatch='//', edgecolor='grey', linewidth=0.5)

        acc = float(r['final_acc']) * 100
        ax.set_title(f'{taxa}\nacc={acc:.1f}%', fontsize=8, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        if col == 0:
            ax.set_ylabel(model_name, fontsize=9, fontweight='bold')

# shared legend
legend_patches = [plt.Rectangle((0,0),1,1, color=c) for c in CLASS_COLORS]
fig.legend(legend_patches, CLASSES, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.savefig('ood_per_class_f1.png', dpi=150, bbox_inches='tight')
print("Saved: ood_per_class_f1.png")


# ── Figure 2: Per-class F1 heatmap — models vs folds ─────────────────────────
# One heatmap per class

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 0.8 + 0.6 * n_models), constrained_layout=True)
fig2.suptitle('OOD F1 Heatmap: Models × Held-Out Taxa', fontsize=13, fontweight='bold')

model_names = [m for m, _ in available]

for ci, cls_name in enumerate(CLASSES):
    ax = axes2[ci]
    matrix = np.zeros((n_models, 5))
    matrix[:] = np.nan
    for ri, (_, results) in enumerate(available):
        for fi, taxa in enumerate(TAXA_LIST):
            r = results.get(taxa, results.get(taxa.replace(' ', '_')))
            if r is None: continue
            f1s = f1_per_class(r['preds'], r.get('targets', r.get('trues', [])))
            matrix[ri, fi] = f1s[ci] if not np.isnan(f1s[ci]) else np.nan

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels([t.replace(' ', '\n') for t in TAXA_LIST], fontsize=8)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title(f'{cls_name} F1', fontsize=11, fontweight='bold')

    for ri in range(n_models):
        for fi in range(5):
            v = matrix[ri, fi]
            label = f'{v:.2f}' if not np.isnan(v) else 'N/A'
            ax.text(fi, ri, label, ha='center', va='center', fontsize=8,
                    color='white' if (not np.isnan(v) and v < 0.5) else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.savefig('ood_per_class_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: ood_per_class_heatmap.png")

print("\n=== Done ===")
