#!/usr/bin/env python3
"""Plot OOD results from ood_results.pkl — train/val loss and accuracy per fold."""

import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAXA_LIST   = ['Snake', 'Cone Snail', 'Spider', 'Scorpion', 'Sea Anemone']
COLORS      = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple', 'tomato']

with open('ood_results.pkl', 'rb') as f:
    results = pickle.load(f)

# ── Figure 1: Train vs Val Loss per fold ─────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
fig.suptitle('Multimodal GNN — Taxa OOD: Train vs Val Loss per Fold\n'
             '(no early stopping — run superseded by v2 with early stopping + class weights)',
             fontsize=11, fontweight='bold')

for ax, taxa, col in zip(axes, TAXA_LIST, COLORS):
    r  = results[taxa]
    ep = range(1, len(r['train_losses']) + 1)
    ax.plot(ep, r['train_losses'], color=col,    label='Train', linewidth=1.5)
    ax.plot(ep, r['val_losses'],   color=col, linestyle='--', alpha=0.6, label='Val (OOD)', linewidth=1.5)
    ax.set_title(f'Test: {taxa}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ood_loss_curves_v1.png', dpi=150, bbox_inches='tight')
print("Saved: ood_loss_curves_v1.png")

# ── Figure 2: Train vs Val Accuracy per fold ──────────────────────────────────
fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
fig2.suptitle('Multimodal GNN — Taxa OOD: Train vs Val Accuracy per Fold\n'
              '(note: val acc peaks early then collapses — fixed in v2 with early stopping)',
              fontsize=11, fontweight='bold')

for ax, taxa, col in zip(axes2, TAXA_LIST, COLORS):
    r    = results[taxa]
    ep   = range(1, len(r['train_accs']) + 1)
    best = max(r['val_accs']) * 100
    ax.plot(ep, [a*100 for a in r['train_accs']], color=col,    label='Train', linewidth=1.5)
    ax.plot(ep, [a*100 for a in r['val_accs']],   color=col, linestyle='--', alpha=0.6, label='Val (OOD)', linewidth=1.5)
    ax.axhline(best, color='black', linestyle=':', linewidth=1,
               label=f'Best val: {best:.1f}%')
    ax.axhline(r['final_acc']*100, color='red', linestyle=':', linewidth=1,
               label=f'Final: {r["final_acc"]*100:.1f}%')
    ax.set_title(f'Test: {taxa}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ood_accuracy_curves_v1.png', dpi=150, bbox_inches='tight')
print("Saved: ood_accuracy_curves_v1.png")

# ── Figure 3: Summary bar chart ───────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
final_accs = [results[t]['final_acc']*100 for t in TAXA_LIST]
best_accs  = [max(results[t]['val_accs'])*100 for t in TAXA_LIST]
x = np.arange(len(TAXA_LIST))
w = 0.35
bars1 = ax3.bar(x - w/2, best_accs,  width=w, color='steelblue', label='Best val acc (with early stopping this would be final)')
bars2 = ax3.bar(x + w/2, final_accs, width=w, color='tomato',    label='Final acc (epoch 50, overfitted)')
ax3.axhline(np.mean(best_accs),  color='steelblue', linestyle='--', linewidth=1.5, label=f'Mean best: {np.mean(best_accs):.1f}%')
ax3.axhline(np.mean(final_accs), color='tomato',    linestyle='--', linewidth=1.5, label=f'Mean final: {np.mean(final_accs):.1f}%')
for bar, val in zip(bars1, best_accs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, final_accs):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(TAXA_LIST, fontsize=10)
ax3.set_ylabel('OOD Accuracy (%)'); ax3.set_ylim(0, 105)
ax3.set_title('OOD Accuracy: Best vs Final (shows cost of no early stopping)', fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('ood_accuracy_summary_v1.png', dpi=150, bbox_inches='tight')
print("Saved: ood_accuracy_summary_v1.png")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Fold':<15} {'Best Val':>10} {'Final':>10} {'Lost':>8}")
print("-" * 45)
for taxa in TAXA_LIST:
    r     = results[taxa]
    best  = max(r['val_accs'])*100
    final = r['final_acc']*100
    print(f"{taxa:<15} {best:>9.1f}%  {final:>9.1f}%  {best-final:>6.1f}%")
print("-" * 45)
print(f"{'Mean best':<15} {np.mean(best_accs):>9.1f}%")
print(f"{'Mean final':<15} {np.mean(final_accs):>9.1f}%")
print(f"{'Gain w/ ES':<15} {np.mean(best_accs)-np.mean(final_accs):>9.1f}%")
