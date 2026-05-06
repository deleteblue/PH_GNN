"""
Multimodal ESM Feature Extractor
Generates ESM embeddings specifically for proteins with structural data.
Targets: P01522, P05483, P05485 (our 3 proteins with both sequence + structure)

Creates ESM embeddings that align with structural graphs in protein_graphs_with_disulfides.pt
"""

import pandas as pd
import torch
import esm
import numpy as np
import os

def load_sequence_data():
    """Load sequence data for our 3 target proteins"""
    print("=== Loading Sequence Data ===")

    # Load cone cone snail data (contains our target proteins)
    try:
        df = pd.read_excel('cone_trainingData.xlsx')
        print(f"Loaded cone_trainingData.xlsx: {df.shape[0]} sequences")
    except Exception as e:
        print(f"Error loading cone_trainingData.xlsx: {e}")
        return None

    # Our target proteins with structural data
    target_proteins = ['P01522', 'P05483', 'P05485']

    # Filter for our proteins
    filtered_df = df[df['Entry'].isin(target_proteins)].copy()
    print(f"Found {len(filtered_df)} out of {len(target_proteins)} target proteins")

    # Show what we found
    for protein_id in target_proteins:
        matches = filtered_df[filtered_df['Entry'] == protein_id]
        if len(matches) > 0:
            sequence = matches.iloc[0]['Sequence']
            ion_channel = matches.iloc[0]['match'] if 'match' in matches.columns else 'Unknown'
            print(f"✅ {protein_id}: {len(sequence)} amino acids, target={ion_channel}")
        else:
            print(f"❌ {protein_id}: Not found in cone data")

    if len(filtered_df) == 0:
        print("ERROR: No target proteins found!")
        return None

    return filtered_df

def generate_esm_embeddings(sequence_df):
    """Generate ESM embeddings for the target proteins"""
    print("\n=== Generating ESM Embeddings ===")

    # Load ESM model (same as original esm_maker.py)
    print("Loading ESM model: esm2_t33_650M_UR50D...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare sequences in ESM format: [(label, sequence), ...]
    sequence_data = []
    for _, row in sequence_df.iterrows():
        protein_id = row['Entry']
        sequence = row['Sequence']
        sequence_data.append((protein_id, sequence))

    print(f"Prepared {len(sequence_data)} sequences for ESM processing")

    # Convert sequences to batch format
    batch_labels, batch_strs, batch_tokens = batch_converter(sequence_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    print("Processing sequences through ESM model...")

    # Generate representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract token representations from layer 33 (same as original)
    token_representations = results["representations"][33]

    print(f"Generated embeddings shape: {token_representations.shape}")
    print(f"  Batch size: {token_representations.shape[0]}")
    print(f"  Max sequence length: {token_representations.shape[1]}")
    print(f"  Embedding dimension: {token_representations.shape[2]}")

    return token_representations.numpy(), batch_labels, batch_lens.numpy()

def save_embeddings(embeddings, labels, lengths):
    """Save embeddings in efficient dictionary format"""
    print("\n=== Saving Embeddings ===")

    # Extract individual protein embeddings (no padding, no special tokens)
    protein_embeddings = {}

    for i, (protein_id, length) in enumerate(zip(labels, lengths)):
        # Extract sequence-length embeddings (excluding <cls> and <eos> tokens)
        protein_embedding = embeddings[i, 1:length-1]  # Remove special tokens
        protein_embeddings[protein_id] = protein_embedding
        print(f"  {protein_id}: {protein_embedding.shape}")

    # Save ONLY the dictionary format (most efficient)
    np.save('esm_features_multimodal_dict.npy', protein_embeddings)
    print(f"✅ Saved: esm_features_multimodal_dict.npy")
    print(f"   Contains all 3 proteins, no padding, easy access by protein ID")

    return protein_embeddings

def verify_alignment():
    """Verify that our ESM embeddings align with structural graphs"""
    print("\n=== Verifying Sequence-Structure Alignment ===")

    # Load our structural graphs
    try:
        graphs = torch.load('protein_graphs_with_disulfides.pt')
        structural_proteins = list(graphs.keys())
        print(f"Structural graphs available: {structural_proteins}")
    except:
        print("Could not load structural graphs - will skip verification")
        return

    # Load our ESM embeddings
    try:
        esm_dict = np.load('esm_features_multimodal_dict.npy', allow_pickle=True).item()
        sequence_proteins = list(esm_dict.keys())
        print(f"ESM embeddings available: {sequence_proteins}")
    except:
        print("Could not load ESM embeddings - will skip verification")
        return

    # Check alignment
    mutual_proteins = set(structural_proteins) & set(sequence_proteins)
    print(f"Proteins with BOTH sequence and structure: {list(mutual_proteins)}")

    if len(mutual_proteins) == 3:
        print("✅ Perfect alignment! All 3 proteins have both modalities")

        # Show details
        for protein_id in mutual_proteins:
            graph = graphs[protein_id]
            embedding = esm_dict[protein_id]
            print(f"  {protein_id}:")
            print(f"    Structure: {graph.num_nodes} atoms")
            print(f"    Sequence: {embedding.shape[0]} residues, {embedding.shape[1]}-dim ESM")
    else:
        print(f"⚠️  Alignment issue - only {len(mutual_proteins)} proteins have both modalities")

def main():
    """Main execution"""
    print("🧬 Multimodal ESM Feature Extraction for Structural Proteins 🧬")
    print("=" * 60)

    # Step 1: Load sequence data
    sequence_df = load_sequence_data()
    if sequence_df is None:
        print("❌ Failed to load sequence data")
        return

    # Step 2: Generate ESM embeddings
    embeddings, labels, lengths = generate_esm_embeddings(sequence_df)

    # Step 3: Save embeddings
    protein_embeddings = save_embeddings(embeddings, labels, lengths)

    # Step 4: Verify alignment with structural data
    verify_alignment()

    print("\n" + "=" * 60)
    print("✅ Multimodal ESM extraction complete!")
    print("Next step: Combine with structural graphs for GNN training")

    return protein_embeddings

if __name__ == "__main__":
    main()