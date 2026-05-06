"""
Disulfide Bond Detection with PyTorch Geometric
Builds protein graphs using PyG Data structures for direct GNN compatibility.

Option B Implementation: Mark disulfide bonds as special edge types
- Uses torch_geometric.data.Data for seamless GNN training
- Edge types: spatial_contact, disulfide_bond (special)
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import os

def parse_pdb_to_tensors(pdb_path: str):
    """
    Parse PDB file directly to PyTorch tensors
    Returns: node_features, coordinates, atom_info
    """
    atoms_data = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # Parse PDB ATOM line
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip() or atom_name[0]

                atoms_data.append({
                    'atom_id': atom_id,
                    'atom_name': atom_name,
                    'residue_name': residue_name,
                    'residue_num': residue_num,
                    'x': x, 'y': y, 'z': z,
                    'element': element
                })

    # Convert to tensors
    coordinates = torch.tensor([[atom['x'], atom['y'], atom['z']] for atom in atoms_data], dtype=torch.float)

    # Simple node features: one-hot encoding of element type
    unique_elements = list(set(atom['element'] for atom in atoms_data))
    element_to_idx = {elem: i for i, elem in enumerate(unique_elements)}

    node_features = torch.zeros(len(atoms_data), len(unique_elements))
    for i, atom in enumerate(atoms_data):
        element_idx = element_to_idx[atom['element']]
        node_features[i, element_idx] = 1.0

    return node_features, coordinates, atoms_data, unique_elements

def find_disulfide_bonds(atoms_data: List[Dict], coordinates: torch.Tensor, threshold: float = 2.5):
    """
    Find disulfide bonds between cysteine sulfur atoms
    Returns: List of (idx1, idx2) pairs forming disulfide bonds
    """
    # Find cysteine sulfur indices
    cys_sulfur_indices = []
    for i, atom in enumerate(atoms_data):
        if atom['residue_name'] == 'CYS' and atom['atom_name'] == 'SG':
            cys_sulfur_indices.append(i)

    print(f"Found {len(cys_sulfur_indices)} cysteine sulfur atoms:")
    for i, idx in enumerate(cys_sulfur_indices):
        res_num = atoms_data[idx]['residue_num']
        print(f"  CYS {i+1}: Residue {res_num}, Index {idx}")

    # Find disulfide bonds
    disulfide_bonds = []
    for i in range(len(cys_sulfur_indices)):
        for j in range(i + 1, len(cys_sulfur_indices)):
            idx1, idx2 = cys_sulfur_indices[i], cys_sulfur_indices[j]

            # Calculate distance
            coord1 = coordinates[idx1]
            coord2 = coordinates[idx2]
            distance = torch.norm(coord1 - coord2).item()

            res1 = atoms_data[idx1]['residue_num']
            res2 = atoms_data[idx2]['residue_num']
            print(f"Distance CYS{res1}-SG to CYS{res2}-SG: {distance:.2f}Å")

            if distance <= threshold:
                disulfide_bonds.append((idx1, idx2))
                print(f"  -> DISULFIDE BOND DETECTED!")

    return disulfide_bonds

def build_spatial_edges(coordinates: torch.Tensor, disulfide_pairs: List[Tuple], threshold: float = 4.0):
    """
    Build spatial contact edges (excluding disulfide bonds)
    Returns: edge_index tensor [2, num_edges]
    """
    num_atoms = coordinates.shape[0]
    spatial_edges = []

    # Convert disulfide pairs to set for fast lookup
    disulfide_set = set(disulfide_pairs) | set((j, i) for i, j in disulfide_pairs)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Skip if this is a disulfide bond
            if (i, j) in disulfide_set:
                continue

            # Calculate distance
            distance = torch.norm(coordinates[i] - coordinates[j]).item()
            if distance <= threshold:
                spatial_edges.extend([(i, j), (j, i)])  # Undirected graph

    if len(spatial_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(spatial_edges, dtype=torch.long).t()

def build_disulfide_edges(disulfide_bonds: List[Tuple]):
    """
    Build disulfide bond edges
    Returns: edge_index tensor [2, num_edges]
    """
    if len(disulfide_bonds) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Make undirected
    disulfide_edges = []
    for i, j in disulfide_bonds:
        disulfide_edges.extend([(i, j), (j, i)])

    return torch.tensor(disulfide_edges, dtype=torch.long).t()

def create_protein_graph(pdb_path: str, protein_id: str) -> Data:
    """
    Create PyTorch Geometric Data object for protein
    """
    print(f"\n=== Building PyG graph for {protein_id} ===")
    print(f"PDB file: {pdb_path}")

    # Parse PDB to tensors
    node_features, coordinates, atoms_data, unique_elements = parse_pdb_to_tensors(pdb_path)
    print(f"Parsed {len(atoms_data)} atoms")
    print(f"Elements found: {unique_elements}")

    # Find disulfide bonds
    disulfide_bonds = find_disulfide_bonds(atoms_data, coordinates)

    # Build edges
    spatial_edge_index = build_spatial_edges(coordinates, disulfide_bonds)
    disulfide_edge_index = build_disulfide_edges(disulfide_bonds)

    # Create PyG Data object
    data = Data(
        x=node_features,           # Node features [num_nodes, num_features]
        pos=coordinates,           # 3D coordinates [num_nodes, 3]
        edge_index_spatial=spatial_edge_index,      # Spatial edges [2, num_spatial_edges]
        edge_index_disulfide=disulfide_edge_index,  # Disulfide edges [2, num_disulfide_edges]
        protein_id=protein_id,
        num_nodes=len(atoms_data)
    )

    print(f"Graph created:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Spatial edges: {data.edge_index_spatial.shape[1]}")
    print(f"  Disulfide edges: {data.edge_index_disulfide.shape[1]}")
    print(f"  Node features shape: {data.x.shape}")

    return data

def main():
    """Process our 3 mutual proteins"""

    # Check PyTorch Geometric installation
    try:
        import torch_geometric
        print(f"Using PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("ERROR: PyTorch Geometric not installed!")
        print("Install with: pip install torch_geometric")
        return

    proteins = {
        'P01522': 'conotoxin_structures/P01522_ranked_0.pdb',
        'P05483': 'conotoxin_structures_2/P05483_ranked_0.pdb',
        'P05485': 'conotoxin_structures_2/P05485_ranked_0.pdb'
    }

    protein_graphs = {}

    for protein_id, pdb_path in proteins.items():
        if os.path.exists(pdb_path):
            graph = create_protein_graph(pdb_path, protein_id)
            protein_graphs[protein_id] = graph
        else:
            print(f"ERROR: PDB file not found: {pdb_path}")

    # Summary
    print(f"\n=== FINAL SUMMARY ===")
    for protein_id, graph in protein_graphs.items():
        disulfide_count = graph.edge_index_disulfide.shape[1] // 2  # Undirected
        spatial_count = graph.edge_index_spatial.shape[1] // 2
        print(f"{protein_id}:")
        print(f"  Atoms: {graph.num_nodes}")
        print(f"  Disulfide bonds: {disulfide_count}")
        print(f"  Spatial contacts: {spatial_count}")

    # Save graphs for later use
    torch.save(protein_graphs, 'protein_graphs_with_disulfides.pt')
    print(f"\nGraphs saved to: protein_graphs_with_disulfides.pt")

    return protein_graphs

if __name__ == "__main__":
    graphs = main()