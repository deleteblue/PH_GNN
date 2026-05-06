#!/usr/bin/env python3
"""
Comprehensive ColabFold Structure Generator for ALL Toxin Datasets
Generates 3D structures for:
1. cone training data (145 proteins, cone snail)
2. toxinCombined_Finalv3 (1342 proteins, ion channel-targeting)
3. nonIonToxinCombined (2547 proteins, non ion channel-targeting)
Total: 4034 protein structures
"""

import os
import sys
import pandas as pd
from Bio.SeqIO import write
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import subprocess
import tempfile
from pathlib import Path
import time

def setup_output_directories():
    """Create output directories for all datasets"""
    datasets = {
        'cone': 'cone_structures',
        'toxin_combined': 'toxin_combined_structures',
        'non_ion_toxin': 'non_ion_toxin_structures'
    }

    output_dirs = {}
    for key, dirname in datasets.items():
        output_dir = Path(dirname)
        output_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {output_dir}")
        output_dirs[key] = output_dir

    return output_dirs

def load_all_datasets():
    """Load sequences from all three datasets"""
    datasets = {}

    # 1. cone training data (cone snail)
    print("\n=== Loading cone_trainingData.xlsx ===")
    df_cone = pd.read_excel('cone_trainingData.xlsx')
    datasets['cone'] = {
        'df': df_cone,
        'seq_col': 'Sequence',
        'id_col': 'Entry',
        'class_cols': ['match', 'Zhang result']
    }
    print(f"cone: {len(df_cone)} proteins")
    print("Classes (match):", df_cone['match'].value_counts().to_dict())

    # 2. ToxinCombined (ion channel-targeting)
    print("\n=== Loading toxinCombined_Finalv3.xlsx ===")
    df_toxin = pd.read_excel('toxinCombined_Finalv3.xlsx')
    datasets['toxin_combined'] = {
        'df': df_toxin,
        'seq_col': 'sequence',
        'id_col': 'entry',
        'class_cols': ['ionChannelClass']
    }
    print(f"ToxinCombined: {len(df_toxin)} proteins")
    print("Classes:", df_toxin['ionChannelClass'].value_counts().to_dict())

    # 3. NonIonToxinCombined (non ion channel-targeting)
    print("\n=== Loading nonIonToxinCombined.xlsx ===")
    df_nonion = pd.read_excel('nonIonToxinCombined.xlsx')
    datasets['non_ion_toxin'] = {
        'df': df_nonion,
        'seq_col': 'sequence',
        'id_col': 'entry',
        'class_cols': []  # Non ion channel data doesn't have ion channel classes
    }
    print(f"NonIonToxin: {len(df_nonion)} proteins")

    total_proteins = len(df_cone) + len(df_toxin) + len(df_nonion)
    print(f"\n🧬 TOTAL PROTEINS: {total_proteins}")

    return datasets

def write_fasta_sequence(protein_id, sequence, output_dir, description="Toxin protein"):
    """Write a single sequence to FASTA format"""
    fasta_file = output_dir / f"{protein_id}.fasta"

    # Create SeqRecord
    record = SeqRecord(
        Seq(sequence),
        id=protein_id,
        description=f"{description} {protein_id}"
    )

    # Write FASTA file
    with open(fasta_file, 'w') as f:
        write(record, f, 'fasta')

    return fasta_file

def run_colabfold_batch(fasta_file, output_dir, protein_id, timeout=1800):
    """Run ColabFold batch prediction for a single protein"""
    try:
        # Use just the filename since we'll run from within the output directory
        fasta_filename = fasta_file.name

        cmd = [
            'colabfold_batch',
            fasta_filename,
            '.',  # Current directory (we're already in output_dir)
            '--num-models', '1',
            '--num-recycle', '3',
            '--model-type', 'alphafold2_ptm',
            '--max-msa', '256:512'
        ]

        print(f"Running ColabFold for {protein_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=output_dir
        )

        if result.returncode == 0:
            print(f"✅ ColabFold success for {protein_id}")
            return True
        else:
            print(f"❌ ColabFold failed for {protein_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ ColabFold timeout for {protein_id}")
        return False
    except Exception as e:
        print(f"❌ ColabFold error for {protein_id}: {e}")
        return False

def log_failed_protein(protein_id, sequence, output_dir, reason="ColabFold failed"):
    """Log failed protein to failure log file"""
    failure_log = output_dir / "failed_proteins.txt"

    with open(failure_log, 'a') as f:
        f.write(f"{protein_id}\t{reason}\tLength:{len(sequence)}\n")

    return failure_log

def process_dataset(dataset_key, dataset_info, output_dirs, test_mode=False, test_limit=3):
    """Process a single dataset for structure generation"""
    df = dataset_info['df']
    seq_col = dataset_info['seq_col']
    id_col = dataset_info['id_col']
    output_dir = output_dirs[dataset_key]

    print(f"\n🚀 Processing {dataset_key.upper()} dataset...")

    if test_mode:
        df = df.head(test_limit)
        print(f"🧪 TEST MODE: Processing first {test_limit} sequences")

    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        protein_id = row[id_col]
        sequence = row[seq_col]

        if pd.isna(sequence) or len(str(sequence)) < 10:
            print(f"⚠️ Skipping {protein_id}: Invalid sequence")
            continue

        print(f"[{idx+1}/{len(df)}] Processing {protein_id} (Length: {len(sequence)} AA)")

        # Write FASTA
        fasta_file = write_fasta_sequence(protein_id, sequence, output_dir)

        # Adaptive timeout based on sequence length
        # Longer sequences need more time (>400 AA = 60min, else 30min)
        timeout_duration = 3600 if len(sequence) > 400 else 1800
        timeout_label = "60min" if len(sequence) > 400 else "30min"
        print(f"   Timeout: {timeout_label} ({timeout_duration}s) for {len(sequence)} AA sequence")

        # Run ColabFold with adaptive timeout
        success = run_colabfold_batch(fasta_file, output_dir, protein_id, timeout=timeout_duration)

        if success:
            successful += 1
        else:
            # Just log the failure - no placeholder files
            log_failed_protein(protein_id, sequence, output_dir)
            failed += 1

    print(f"\n📊 {dataset_key.upper()} RESULTS:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    return successful, failed

def generate_fasta_only(dataset_key, dataset_info, output_dirs):
    """Generate only FASTA files without running ColabFold"""
    df = dataset_info['df']
    seq_col = dataset_info['seq_col']
    id_col = dataset_info['id_col']
    output_dir = output_dirs[dataset_key]

    print(f"\n📝 Generating FASTA files for {dataset_key.upper()} dataset...")

    successful = 0
    skipped = 0

    for idx, row in df.iterrows():
        protein_id = row[id_col]
        sequence = row[seq_col]

        if pd.isna(sequence) or len(str(sequence)) < 10:
            print(f"⚠️ Skipping {protein_id}: Invalid sequence")
            skipped += 1
            continue

        # Check if FASTA already exists
        fasta_file = output_dir / f"{protein_id}.fasta"
        if fasta_file.exists():
            continue

        print(f"[{idx+1}/{len(df)}] Creating FASTA for {protein_id} (Length: {len(sequence)} AA)")

        # Write FASTA file only
        write_fasta_sequence(protein_id, sequence, output_dir)
        successful += 1

    print(f"\n📊 {dataset_key.upper()} FASTA GENERATION RESULTS:")
    print(f"✅ FASTA files created: {successful}")
    print(f"⚠️ Skipped (invalid): {skipped}")

    return successful, skipped

def main():
    """Main execution function"""
    # Check for FASTA-only mode
    fasta_only = len(sys.argv) > 1 and sys.argv[1] == 'fasta_only'
    test_mode = len(sys.argv) > 1 and sys.argv[1] == 'test'
    test_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if fasta_only:
        print("📝 FASTA ONLY MODE - Generating FASTA files for external GPU processing")
    elif test_mode:
        print("🧪 TEST MODE - Processing limited sequences from each dataset")
        print(f"Limit: {test_limit} sequences per dataset")
    else:
        print("🚀 FULL MODE - Processing ALL sequences from all datasets")

    # Setup directories
    output_dirs = setup_output_directories()

    if fasta_only:
        # Load only non_ion_toxin dataset
        print("\n=== Loading nonIonToxinCombined.xlsx ===")
        df_nonion = pd.read_excel('nonIonToxinCombined.xlsx')
        dataset_info = {
            'df': df_nonion,
            'seq_col': 'sequence',
            'id_col': 'entry',
            'class_cols': []
        }
        print(f"NonIonToxin: {len(df_nonion)} proteins")

        # Generate FASTA files only
        successful, skipped = generate_fasta_only('non_ion_toxin', dataset_info, output_dirs)

        # Final summary
        print(f"\n{'='*60}")
        print("📝 FASTA FILE GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ FASTA files created: {successful}")
        print(f"⚠️ Skipped (invalid): {skipped}")
        print(f"📁 Output directory: {output_dirs['non_ion_toxin']}")
        print("📧 Ready for external GPU processing!")
        return

    # Load all datasets
    datasets = load_all_datasets()

    # Process each dataset
    total_successful = 0
    total_failed = 0

    for dataset_key, dataset_info in datasets.items():
        # Skip cone dataset - already completed
        if dataset_key == 'cone':
            print(f"\n⏭️ SKIPPING {dataset_key.upper()} dataset - already completed")
            continue

        # Skip toxin_combined dataset - already completed in Phase 1
        if dataset_key == 'toxin_combined':
            print(f"\n⏭️ SKIPPING {dataset_key.upper()} dataset - already completed (1,243 structures generated)")
            continue

        # Process non_ion_toxin dataset - Phase 2
        # if dataset_key == 'non_ion_toxin':
        #     print(f"\n⏭️ SKIPPING {dataset_key.upper()} dataset - will process separately")
        #     continue

        successful, failed = process_dataset(
            dataset_key, dataset_info, output_dirs, test_mode, test_limit
        )
        total_successful += successful
        total_failed += failed

    # Final summary
    print(f"\n{'='*60}")
    print("🎯 COMPREHENSIVE STRUCTURE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Total Successful: {total_successful}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📁 Output directories:")
    for key, output_dir in output_dirs.items():
        print(f"   {key}: {output_dir}")

if __name__ == "__main__":
    main()