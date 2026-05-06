#!/usr/bin/env python3
"""
Persistent Homology Analysis for Toxin Ion Channel Classification
=================================================================

This script applies persistent homology to analyze 3D protein structures
and correlate topological features with ion channel targeting specificity.

Mathematical Approach:
- Alpha complexes to capture protein topology
- Persistent homology to track holes across scales
- Betti numbers to quantify topological features
- Comparison across ion channel classes

Target Proteins:
- A3DT44: Other/K class (conflict between studies)
- A6YR20: Na-targeting
- B3SVF0: K-targeting

Novel Hypothesis: Different ion channel targeting mechanisms require
distinct protein cavity topologies that persist across multiple scales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GUDHI for persistent homology
import gudhi

def extract_pdb_coordinates(pdb_path, atom_types=['CA', 'CB', 'N', 'C', 'O']):
    """
    Extract 3D atomic coordinates from PDB file.

    Args:
        pdb_path: Path to PDB file
        atom_types: List of atom types to extract (default: backbone + CB)

    Returns:
        coordinates: numpy array of 3D coordinates (n_atoms, 3)
        atom_info: list of dictionaries with atom information
    """
    coordinates = []
    atom_info = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name in atom_types:
                    # Extract coordinates
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    # Store coordinate and metadata
                    coordinates.append([x, y, z])
                    atom_info.append({
                        'atom_name': atom_name,
                        'residue': line[17:20].strip(),
                        'chain': line[21:22],
                        'residue_num': int(line[22:26])
                    })

    return np.array(coordinates), atom_info

def compute_persistent_homology(coordinates, max_dimension=2, max_edge_length=20.0):
    """
    Compute persistent homology using GUDHI Alpha Complex.

    Args:
        coordinates: numpy array of 3D coordinates
        max_dimension: maximum homological dimension to compute
        max_edge_length: maximum edge length for alpha complex

    Returns:
        persistence: persistent homology pairs
        alpha_complex: GUDHI alpha complex object
        simplex_tree: GUDHI simplex tree
    """
    print(f"Computing persistent homology for {len(coordinates)} atoms")

    # Create alpha complex
    alpha_complex = gudhi.AlphaComplex(points=coordinates)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=max_edge_length**2)

    # Compute persistent homology
    persistence = simplex_tree.persistence(
        homology_coeff_field=2,  # Use Z/2Z coefficients
        min_persistence=0.1      # Filter out very short-lived features
    )

    return persistence, alpha_complex, simplex_tree

def extract_topological_features(persistence):
    """
    Extract numerical features from persistence diagrams.

    Args:
        persistence: GUDHI persistence pairs

    Returns:
        features: dictionary of topological features
    """
    features = {
        'betti_0_births': [],      # Connected component births
        'betti_0_deaths': [],      # Connected component deaths
        'betti_0_lifespans': [],   # Component persistence
        'betti_1_births': [],      # Loop births
        'betti_1_deaths': [],      # Loop deaths
        'betti_1_lifespans': [],   # Loop persistence
        'betti_2_births': [],      # Void births
        'betti_2_deaths': [],      # Void deaths
        'betti_2_lifespans': [],   # Void persistence
    }

    for dimension, (birth, death) in persistence:
        lifespan = death - birth if death != float('inf') else 50.0  # Cap infinite persistence

        if dimension == 0:  # Connected components (Betti-0)
            features['betti_0_births'].append(birth)
            features['betti_0_deaths'].append(death if death != float('inf') else 50.0)
            features['betti_0_lifespans'].append(lifespan)

        elif dimension == 1:  # Loops (Betti-1)
            features['betti_1_births'].append(birth)
            features['betti_1_deaths'].append(death if death != float('inf') else 50.0)
            features['betti_1_lifespans'].append(lifespan)

        elif dimension == 2:  # Voids (Betti-2)
            features['betti_2_births'].append(birth)
            features['betti_2_deaths'].append(death if death != float('inf') else 50.0)
            features['betti_2_lifespans'].append(lifespan)

    # Compute summary statistics
    for dim in [0, 1, 2]:
        lifespans = features[f'betti_{dim}_lifespans']
        if lifespans:
            features[f'betti_{dim}_count'] = len(lifespans)
            features[f'betti_{dim}_max_lifespan'] = max(lifespans)
            features[f'betti_{dim}_mean_lifespan'] = np.mean(lifespans)
            features[f'betti_{dim}_total_persistence'] = sum(lifespans)
        else:
            features[f'betti_{dim}_count'] = 0
            features[f'betti_{dim}_max_lifespan'] = 0
            features[f'betti_{dim}_mean_lifespan'] = 0
            features[f'betti_{dim}_total_persistence'] = 0

    return features

def plot_persistence_diagram(persistence, title, save_path=None):
    """
    Plot persistence diagram with birth-death coordinates.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Separate by dimension
    dimensions = {}
    for dim, (birth, death) in persistence:
        if dim not in dimensions:
            dimensions[dim] = {'births': [], 'deaths': []}
        dimensions[dim]['births'].append(birth)
        dimensions[dim]['deaths'].append(death if death != float('inf') else 50.0)

    # Plot each dimension with different colors
    colors = ['red', 'blue', 'green']
    labels = ['Connected Components (H₀)', 'Loops (H₁)', 'Voids (H₂)']

    for i, dim in enumerate(sorted(dimensions.keys())):
        if i < len(colors):
            ax.scatter(dimensions[dim]['births'], dimensions[dim]['deaths'],
                      c=colors[i], label=labels[i], alpha=0.7, s=60)

    # Plot diagonal (birth = death line)
    max_val = max([max(dimensions[dim]['deaths']) for dim in dimensions.keys()])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Birth = Death')

    ax.set_xlabel('Birth (Å)', fontsize=12)
    ax.set_ylabel('Death (Å)', fontsize=12)
    ax.set_title(f'Persistence Diagram: {title}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_protein(protein_id, pdb_path, ion_channel_class):
    """
    Complete persistent homology analysis for a single protein.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {protein_id} ({ion_channel_class} class)")
    print(f"PDB File: {pdb_path}")
    print(f"{'='*60}")

    # Extract coordinates
    coordinates, atom_info = extract_pdb_coordinates(pdb_path)
    print(f"Extracted {len(coordinates)} atoms from {len(set([a['residue_num'] for a in atom_info]))} residues")

    # Compute persistent homology
    persistence, alpha_complex, simplex_tree = compute_persistent_homology(coordinates)

    # Extract features
    features = extract_topological_features(persistence)

    # Plot persistence diagram
    plot_title = f"{protein_id} ({ion_channel_class})"
    save_path = f"persistence_diagram_{protein_id}.png"
    plot_persistence_diagram(persistence, plot_title, save_path)

    # Print key topological features
    print(f"\nTOPOLOGICAL FEATURES:")
    print(f"Connected Components (Betti-0): {features['betti_0_count']}")
    print(f"  Max persistence: {features['betti_0_max_lifespan']:.2f}Å")
    print(f"Loops (Betti-1): {features['betti_1_count']}")
    print(f"  Max persistence: {features['betti_1_max_lifespan']:.2f}Å")
    print(f"Voids (Betti-2): {features['betti_2_count']}")
    print(f"  Max persistence: {features['betti_2_max_lifespan']:.2f}Å")

    return {
        'protein_id': protein_id,
        'ion_channel_class': ion_channel_class,
        'coordinates': coordinates,
        'persistence': persistence,
        'features': features,
        'num_atoms': len(coordinates),
        'num_residues': len(set([a['residue_num'] for a in atom_info]))
    }

def main():
    """
    Main analysis pipeline for persistent homology of toxin proteins.
    """
    print("PERSISTENT HOMOLOGY ANALYSIS FOR TOXIN ION CHANNEL CLASSIFICATION")
    print("=" * 80)

    # Define our test proteins
    proteins = [
        {
            'id': 'A3DT44',
            'path': 'cone_structures/A3DT44_Toxin_protein_A3DT44_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb',
            'class': 'Other'  # Note: conflict with Zhang (K), using match result
        },
        {
            'id': 'A6YR20',
            'path': 'cone_structures/A6YR20_Toxin_protein_A6YR20_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb',
            'class': 'Na'
        },
        {
            'id': 'B3SVF0',
            'path': 'cone_structures/B3SVF0_Toxin_protein_B3SVF0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb',
            'class': 'K'
        }
    ]

    # Analyze each protein
    results = []
    for protein in proteins:
        if Path(protein['path']).exists():
            result = analyze_protein(protein['id'], protein['path'], protein['class'])
            results.append(result)
        else:
            print(f"WARNING: PDB file not found: {protein['path']}")

    # Compare topological features across ion channel classes
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS ACROSS ION CHANNEL CLASSES")
    print(f"{'='*80}")

    # Create comparison DataFrame
    comparison_data = []
    for result in results:
        features = result['features']
        comparison_data.append({
            'Protein': result['protein_id'],
            'Class': result['ion_channel_class'],
            'Atoms': result['num_atoms'],
            'Residues': result['num_residues'],
            'Components': features['betti_0_count'],
            'Component_Persistence': features['betti_0_total_persistence'],
            'Loops': features['betti_1_count'],
            'Loop_Persistence': features['betti_1_total_persistence'],
            'Voids': features['betti_2_count'],
            'Void_Persistence': features['betti_2_total_persistence'],
            'Max_Loop_Persistence': features['betti_1_max_lifespan'],
            'Max_Void_Persistence': features['betti_2_max_lifespan']
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nTOPOLOGICAL FEATURE COMPARISON:")
    print(comparison_df.round(2))

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Loop count vs class
    axes[0, 0].bar(comparison_df['Class'], comparison_df['Loops'],
                   color=['orange', 'blue', 'green'])
    axes[0, 0].set_title('Number of Persistent Loops (H₁)')
    axes[0, 0].set_ylabel('Count')

    # Plot 2: Void count vs class
    axes[0, 1].bar(comparison_df['Class'], comparison_df['Voids'],
                   color=['orange', 'blue', 'green'])
    axes[0, 1].set_title('Number of Persistent Voids (H₂)')
    axes[0, 1].set_ylabel('Count')

    # Plot 3: Max loop persistence vs class
    axes[1, 0].bar(comparison_df['Class'], comparison_df['Max_Loop_Persistence'],
                   color=['orange', 'blue', 'green'])
    axes[1, 0].set_title('Maximum Loop Persistence')
    axes[1, 0].set_ylabel('Persistence (Å)')

    # Plot 4: Max void persistence vs class
    axes[1, 1].bar(comparison_df['Class'], comparison_df['Max_Void_Persistence'],
                   color=['orange', 'blue', 'green'])
    axes[1, 1].set_title('Maximum Void Persistence')
    axes[1, 1].set_ylabel('Persistence (Å)')

    plt.tight_layout()
    plt.savefig('topological_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Statistical hypothesis test placeholder
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    print(f"{'='*80}")
    print("Sample size (n=3) is too small for statistical significance testing.")
    print("Key observations for future research:")

    # Identify patterns
    na_protein = comparison_df[comparison_df['Class'] == 'Na'].iloc[0]
    k_protein = comparison_df[comparison_df['Class'] == 'K'].iloc[0]
    other_protein = comparison_df[comparison_df['Class'] == 'Other'].iloc[0]

    print(f"- Na-targeting ({na_protein['Protein']}): {na_protein['Loops']} loops, {na_protein['Voids']} voids")
    print(f"- K-targeting ({k_protein['Protein']}): {k_protein['Loops']} loops, {k_protein['Voids']} voids")
    print(f"- Other-targeting ({other_protein['Protein']}): {other_protein['Loops']} loops, {other_protein['Voids']} voids")

    print(f"\nHYPOTHESIS for expanded dataset:")
    print("Different ion channel targets require distinct cavity topologies")
    print("→ Na/K/Ca channels should show characteristic persistence patterns")
    print("→ Need larger dataset (n>>3) for statistical validation")

    # Save results
    comparison_df.to_csv('persistent_homology_results.csv', index=False)
    print(f"\nResults saved to: persistent_homology_results.csv")

    return results, comparison_df

if __name__ == "__main__":
    results, comparison_df = main()