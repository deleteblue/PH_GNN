# PH-GNN: Persistent Homology Graph Neural Network for Compound Ion Channel Classification

## Overview

PH-GNN is a multimodal deep learning framework for classifying animal compound proteins by ion channel targeting specificity. It integrates three complementary protein representations — sequence embeddings, 3D atomic structure graphs, and persistent homology topology features — into a unified GNN architecture.

**Classification task**: 3-class — Calcium, Potassium, Sodium ion channel compounds  
**Dataset**: 1,373 proteins (Calcium: 216, Potassium: 542, Sodium: 615)  
**Species coverage**: Snake, Cone Snail, Spider, Scorpion, Sea Anemone

---

## Key Results

### Random 80/20 Split (stratified, seed 42)

| Model | Test Acc | Macro F1 |
|---|---|---|
| **PH-GNN (ours)** | **85.8%** | **0.82** |
| Statistical Manifold | 86.2% | 0.84 |
| Spectral GNN | 84.0% | 0.81 |
| Baseline GNN | 82.9% | 0.81 |
| Riemannian GNN | 82.9% | 0.77 |
| Invariant SE(3) GNN | 81.1% | 0.79 |

### Taxa Leave-One-Out OOD Cross-Validation (primary evaluation)

| Model | Snake | Cone Snail | Spider | Scorpion | Sea Anemone | **Mean** |
|---|---|---|---|---|---|---|
| **PH-GNN (ours)** | **75.3%** | 51.2% | **46.7%** | **74.1%** | 69.6% | **63.4%** |
| Invariant SE(3) | 49.4% | **52.0%** | 41.4% | 61.1% | **73.3%** | 55.4% |
| Statistical Manifold | 46.8% | 46.5% | 43.1% | 70.0% | 69.6% | 55.2% |
| Baseline GNN | 45.5% | 52.0% | 43.1% | 66.7% | 68.9% | 55.2% |
| Spectral GNN | 48.1% | 48.8% | 41.2% | 62.8% | 70.8% | 54.3% |
| Riemannian GNN | 45.5% | 52.8% | 41.4% | 61.9% | 68.9% | 54.1% |
| ESM + Random Forest | 45.1% | 45.1% | 24.0% | 6.3% | N/A | ~30% |

OOD evaluation withholds one entire taxa (species group) as the test set and trains on the remaining four, testing true cross-species generalization.

---

## Architecture

### PH-GNN (Persistent Homology GNN)

Three encoders are concatenated and passed to a classifier:

```
Protein PDB → StructuralEncoder  (GAT, 3 layers, 256-dim)  ─┐
Protein seq → ESM-2 (1280-dim)   → SequenceEncoder (256-dim) ├→ [768-dim] → FC → 3-class
PDB Cα atoms → GUDHI Alpha complex → HomologyEncoder (256-dim) ─┘
```

**StructuralEncoder**: Graph Attention Network (GAT, 4 heads) on all-atom spatial contact graph + dedicated disulfide bond attention branch. Node features: 5D element one-hot + 3D coordinates.

**SequenceEncoder**: Linear projection of pre-pooled ESM-2 embeddings (1280 → 256).

**HomologyEncoder**: MLP over 12D persistent homology feature vector (Betti-0/1/2: count, max lifespan, mean lifespan, total persistence computed at scales 0.5–20 Å using GUDHI Alpha complex).

**Fusion**: Concatenation (256 × 3 = 768) → Linear(512) → ReLU → Dropout(0.1) → Linear(3).

---

## Repository Structure

```
# Core architecture
multimodal_gnn_fixed.py             Base GNN: StructuralEncoder + SequenceEncoder

# PH-GNN (main model)
train_homology_multimodal.py        Random-split training
train_ood_homology.py               Taxa leave-one-out OOD cross-validation

# Comparison models
train_full_multimodal.py            Baseline GNN (no topology)
train_full_multimodal_v2.py         Baseline GNN + per-epoch val loss + plots
train_riemannian_multimodal.py      Riemannian metric learning variant
train_spectral_multimodal.py        Spectral graph theory variant
train_invariant_multimodal.py       SE(3)-invariant variant
train_statistical_manifold_multimodal.py  Fisher information manifold variant
train_ood_*.py                      OOD versions of each model above

# Feature engineering
fasta_based_esm_generator.py        ESM-2 embedding extraction (FASTA-aligned)
disulfide_pyg_builder.py            Disulfide bond detection + PyG graph construction
persistent_homology_analysis.py     GUDHI persistent homology feature extraction

# Data preparation
generate_taxa_fastas.py             Group proteins by taxa into per-species FASTA files
deduplicate_taxa_fastas.py          Remove identical sequences across datasets
colabfold_structure_generator.py    ColabFold structure generation pipeline
colabfold_all_datasets.py           Batch structure generation across all datasets
data_alignment.py                   Sequence–structure intersection analysis

# Evaluation & visualisation
multimodal_evaluation.py            ROC curves, confusion matrix, per-class metrics
ablation_study.py                   4-way modality ablation
plot_ood_results.py                 OOD accuracy/loss curves per fold
plot_ood_per_class.py               Per-class F1 heatmaps across models and folds
tsne_umap_all_models.py             t-SNE / UMAP embedding visualisation (all models)
tsne_homology.py                    t-SNE for PH-GNN embeddings
xai_dash_app.py                     Interactive XAI app (UMAP + residue attention + 3D)
extract_xai_data.py                 Extract attention weights for XAI

# Environment
environment.def                     Apptainer container definition
requirements.txt                    Python package versions
environment_1.yml                   Conda environment specification
```

---

## Reproducing Results

### 1. Environment Setup

**Option A — Apptainer container (recommended, includes GUDHI + JAX/CUDA):**
```bash
apptainer build environment.sif environment.def
apptainer run --nv environment.sif
```

**Option B — Conda:**
```bash
conda env create -f environment_1.yml
conda activate py371
pip install -r requirements.txt
```

### 2. Data

Protein structures (PDB files) are required as input. They are not included in this repository due to size. Generate them with ColabFold or AlphaFold2:

```bash
# Generate structures for all proteins
python colabfold_all_datasets.py

# Or use the SLURM script on a cluster
sbatch colabfold_all_full_slurm.sh
```

Expected directory layout after structure generation:
```
cone_structures/          # Cone snail PDB files
compound_combined_structures/ # Ion channel compound PDB files
```

### 3. Feature Extraction

```bash
# ESM-2 sequence embeddings (requires GPU, ~6 hrs on cluster)
sbatch fasta_based_esm_slurm.sh
# Output: esm_features_fasta_aligned.pt

# Persistent homology features are computed on-the-fly during training
# (no pre-computation needed)
```

### 4. Training PH-GNN

**Random 80/20 split:**
```bash
python train_homology_multimodal.py
# or on SLURM:
sbatch train_homology_multimodal_slurm.sh
```

**Taxa leave-one-out OOD cross-validation:**
```bash
python train_ood_homology.py
# or on SLURM:
sbatch train_ood_homology_slurm.sh
```

Training hyperparameters (PH-GNN):
- Optimizer: Adam, lr=0.001, weight_decay=0
- Loss: class-weighted CrossEntropyLoss (weight = 1/class_count, normalized)
- Early stopping: patience=10 on validation loss
- Epochs: up to 50

### 5. Reproducing All Comparison Models

Each model follows the same pattern:
```bash
python train_<model>_multimodal.py        # random split
python train_ood_<model>.py               # OOD cross-validation
```
where `<model>` is one of: `full`, `riemannian`, `spectral`, `invariant`, `statistical_manifold`.

### 6. Evaluation & Figures

```bash
python multimodal_evaluation.py           # ROC + confusion matrix
python ablation_study.py                  # Modality ablation
python plot_ood_results.py                # OOD training curves
python plot_ood_per_class.py              # Per-class F1 heatmap
python tsne_umap_all_models.py            # Embedding visualisation
```

---

## Dependencies

Core requirements: Python 3.7, PyTorch 1.13, PyTorch Geometric 2.3, fair-esm 2.0, GUDHI 3.8, scikit-learn 1.0, pandas, numpy, matplotlib, seaborn.

See `requirements.txt` for exact pinned versions. See `environment.def` for the full Apptainer container spec (includes CUDA 11.7, JAX, RDKit).

---

## Data Sources

- Ion channel compounds: UniProtKB (curated Calcium/Potassium/Sodium annotations)
- Non-ion-channel compounds: UniProtKB (negative class)
- Cone snail compounds: ConoServer
- Protein structures: AlphaFold2 via ColabFold
- Sequence embeddings: ESM-2 (650M parameter model, Meta AI)
