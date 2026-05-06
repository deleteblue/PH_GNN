#!/usr/bin/env python3
"""
FASTA-Based ESM Feature Generator
Perfect 1:1 alignment by reading FASTA files from structure directories.
"""

import pandas as pd
import torch
import esm
import numpy as np
from pathlib import Path
import os

def setup_esm_model():
    """Load ESM-2 model"""
    print("=== Loading ESM-2 Model ===")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device('cpu')
    model = model.to(device)
    print(f"ESM model loaded on: {device}")

    return model, batch_converter, device

def get_structure_fasta_pairs():
    """Get all structure+FASTA pairs from both directories"""
    print("=== Scanning Structure+FASTA Pairs ===")

    pairs = []

    # cone structures
    cone_dir = Path('cone_structures')
    if cone_dir.exists():
        for pdb_file in cone_dir.glob('*.pdb'):
            protein_id = pdb_file.name.split('_')[0]
            fasta_file = cone_dir / f"{protein_id}.fasta"

            if fasta_file.exists():
                pairs.append({
                    'protein_id': protein_id,
                    'dataset': 'cone',
                    'pdb_path': str(pdb_file),
                    'fasta_path': str(fasta_file)
                })
        print(f"cone pairs: {len([p for p in pairs if p['dataset'] == 'cone'])}")

    # ToxinCombined structures
    toxin_dir = Path('toxin_combined_structures')
    if toxin_dir.exists():
        for pdb_file in toxin_dir.glob('*.pdb'):
            protein_id = pdb_file.name.split('_')[0]
            fasta_file = toxin_dir / f"{protein_id}.fasta"

            if fasta_file.exists():
                pairs.append({
                    'protein_id': protein_id,
                    'dataset': 'toxin_combined',
                    'pdb_path': str(pdb_file),
                    'fasta_path': str(fasta_file)
                })
        print(f"ToxinCombined pairs: {len([p for p in pairs if p['dataset'] == 'toxin_combined'])}")

    print(f"Total structure+FASTA pairs: {len(pairs)}")
    return pairs

def read_fasta_sequence(fasta_path):
    """Read sequence from FASTA file"""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()

    # Skip header line, join sequence lines
    sequence = ''.join(line.strip() for line in lines[1:])
    return sequence

def generate_esm_features(model, batch_converter, device, pairs, max_batch_size=10):
    """Generate ESM features from FASTA files"""
    print("=== Generating ESM Features from FASTA Files ===")

    esm_features = {}
    metadata = []

    for i in range(0, len(pairs), max_batch_size):
        batch_num = i // max_batch_size + 1
        total_batches = (len(pairs) + max_batch_size - 1) // max_batch_size
        print(f"Processing batch {batch_num}/{total_batches} ({len(pairs[i:i + max_batch_size])} proteins)")

        batch_pairs = pairs[i:i + max_batch_size]
        batch_data = []

        # Read sequences from FASTA files
        for pair in batch_pairs:
            sequence = read_fasta_sequence(pair['fasta_path'])
            batch_data.append((pair['protein_id'], sequence))

            # Store metadata with compound key
            feature_key = f"{pair['dataset']}_{pair['protein_id']}"
            metadata.append({
                'feature_key': feature_key,
                'protein_id': pair['protein_id'],
                'dataset': pair['dataset'],
                'sequence_length': len(sequence),
                'structure_file': pair['pdb_path'],
                'fasta_file': pair['fasta_path']
            })

        # Convert batch
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != batch_converter.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        # Generate representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        token_representations = results["representations"][33]

        # Extract sequence representations
        for j, pair in enumerate(batch_pairs):
            protein_id = pair['protein_id']
            dataset = pair['dataset']
            # Use compound key to handle duplicates between datasets
            feature_key = f"{dataset}_{protein_id}"
            seq_len = batch_lens[j] - 2  # Exclude <cls> and <eos> tokens
            seq_repr = token_representations[j, 1:seq_len+1].mean(0)  # Mean over sequence length
            esm_features[feature_key] = seq_repr.cpu().numpy()

    return esm_features, metadata

def main():
    """Main execution function"""
    print("=== FASTA-Based ESM Feature Generation ===")

    # Step 1: Get all structure+FASTA pairs
    pairs = get_structure_fasta_pairs()

    # Step 2: Setup ESM model
    model, batch_converter, device = setup_esm_model()

    # Step 3: Generate ESM features from FASTA files
    esm_features, metadata = generate_esm_features(model, batch_converter, device, pairs)

    # Step 4: Final verification
    print("=== Final Verification ===")
    print(f"Structure+FASTA pairs: {len(pairs)}")
    print(f"ESM features generated: {len(esm_features)}")
    print(f"Metadata entries: {len(metadata)}")

    if len(pairs) == len(esm_features) == len(metadata):
        print("✅ PERFECT ALIGNMENT: All counts match!")
    else:
        print("❌ ALIGNMENT ERROR: Counts don't match!")
        return

    # Step 5: Save results
    print("=== Saving Perfect Alignment Results ===")

    # Save ESM features
    torch.save(esm_features, 'esm_features_fasta_aligned.pt')
    print(f"Saved: esm_features_fasta_aligned.pt ({len(esm_features)} proteins)")

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df = metadata_df.reset_index().rename(columns={'index': 'feature_index'})
    metadata_df.to_csv('feature_metadata_fasta_aligned.csv', index=False)
    print(f"Saved: feature_metadata_fasta_aligned.csv")

    # Summary statistics
    print("=== Summary Statistics ===")
    print(f"Total perfectly aligned proteins: {len(esm_features)}")
    dataset_counts = metadata_df['dataset'].value_counts()
    print("\nDataset distribution:")
    print(dataset_counts)

    print(f"\nFeature shape: {next(iter(esm_features.values())).shape}")

    print("\n=== FASTA-BASED ESM FEATURES COMPLETE! ===")
    print("✅ Perfect 1:1 alignment guaranteed by FASTA+structure pairing")
    print("✅ Ready for multimodal GNN training")

if __name__ == "__main__":
    main()