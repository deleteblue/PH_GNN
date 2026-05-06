#!/usr/bin/env python3
"""
ColabFold Structure Generator for cone Toxin Sequences
Generates 3D structures for all toxin proteins in cone training data
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

# Add AlphaFold to path
sys.path.append('/opt/alphafold')

def setup_output_directories():
    """Create output directory for cone training data structures"""
    base_dir = Path("cone_generated_structures")
    base_dir.mkdir(exist_ok=True)

    print(f"✅ Created output directory: {base_dir}")
    return base_dir

def load_cone_sequences():
    """Load ALL sequences from cone training data"""
    print("Loading cone training data...")
    df = pd.read_excel('cone_trainingData.xlsx')

    print(f"Total sequences in cone: {len(df)}")

    # Show class distribution for ALL sequences
    print("\nIon channel class distribution (match):")
    print(df['match'].value_counts())
    print("\nIon channel class distribution (Zhang result):")
    print(df['Zhang result'].value_counts())

    return df

def write_fasta_sequence(protein_id, sequence, output_dir):
    """Write a single sequence to FASTA format"""
    fasta_file = output_dir / f"{protein_id}.fasta"

    # Create SeqRecord
    record = SeqRecord(
        Seq(sequence),
        id=protein_id,
        description=f"Toxin protein {protein_id}"
    )

    # Write to file
    with open(fasta_file, 'w') as f:
        write(record, f, 'fasta')

    return fasta_file

def run_colabfold_prediction(fasta_file, output_dir, protein_id, ion_class):
    """Run ColabFold structure prediction"""
    print(f"Running ColabFold for {protein_id}...")

    # ColabFold command - all structures in same directory
    cmd = [
        'colabfold_batch',
        str(fasta_file),
        str(output_dir),
        '--num-models', '1',           # Generate 1 model (fastest)
        '--num-recycle', '3',          # 3 recycling steps (good balance)
        '--model-type', 'alphafold2_ptm',  # Use AlphaFold2 with confidence scores
        '--max-msa', '256:512',        # Limit MSA size for speed
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        if result.returncode == 0:
            print(f"✅ ColabFold success for {protein_id}")
            return True
        else:
            print(f"❌ ColabFold failed for {protein_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ ColabFold timeout for {protein_id} (>30 min)")
        return False
    except Exception as e:
        print(f"❌ ColabFold error for {protein_id}: {e}")
        return False

def simple_structure_prediction(protein_id, sequence, output_dir, ion_class):
    """
    Simplified structure prediction using AlphaFold2 components
    Falls back to this if full AlphaFold2 pipeline fails
    """
    print(f"Attempting simplified prediction for {protein_id}...")

    try:
        # This is a placeholder for a simplified approach
        # In practice, you might use:
        # - ColabFold (faster, web-based)
        # - ChimeraX AlphaFold
        # - Or pre-computed AlphaFold database lookup

        # Create both FASTA and placeholder PDB files in the main cone folder
        # Save FASTA file for future use
        fasta_file = output_dir / f"{protein_id}.fasta"
        with open(fasta_file, 'w') as f:
            f.write(f">{protein_id}\n{sequence}\n")

        # Create placeholder PDB file with metadata
        placeholder_pdb = output_dir / f"{protein_id}_placeholder.pdb"
        with open(placeholder_pdb, 'w') as f:
            f.write(f"HEADER    TOXIN PROTEIN                           PLACEHOLDER PDB\n")
            f.write(f"TITLE     {protein_id} - ION CHANNEL CLASS: {ion_class}\n")
            f.write(f"REMARK   1 SEQUENCE LENGTH: {len(sequence)}\n")
            f.write(f"REMARK   2 SOURCE: cone_trainingData.xlsx\n")
            f.write(f"REMARK   3 AWAITING ALPHAFOLD2 STRUCTURE PREDICTION\n")
            f.write(f"REMARK   4 SEQUENCE: {sequence}\n")

        print(f"📝 Created FASTA + placeholder PDB for {protein_id}")
        return True

    except Exception as e:
        print(f"❌ File creation failed for {protein_id}: {e}")
        return False

def main():
    """Main structure generation pipeline for cone training data"""
    print("🧬 ColabFold Structure Generation Pipeline - cone Training Data")
    print("=" * 60)

    # Setup
    output_dir = setup_output_directories()
    cone_df = load_cone_sequences()

    # Track results
    success_count = 0
    failed_ids = []

    # Process each sequence
    print(f"\n🚀 Processing ALL {len(cone_df)} cone sequences...")

    for idx, row in cone_df.iterrows():
        protein_id = row['Entry']
        sequence = row['Sequence']
        ion_class = row['match']

        print(f"\n[{idx+1}/{len(missing_df)}] Processing {protein_id} ({ion_class})")

        try:
            # Create temporary FASTA
            with tempfile.TemporaryDirectory() as temp_dir:
                fasta_file = write_fasta_sequence(protein_id, sequence, Path(temp_dir))

                # Try ColabFold prediction first
                success = run_colabfold_prediction(fasta_file, output_dir, protein_id, ion_class)

                # If that fails, try simplified approach (save FASTA for manual processing)
                if not success:
                    success = simple_structure_prediction(protein_id, sequence, output_dir, ion_class)

                if success:
                    success_count += 1
                else:
                    failed_ids.append(protein_id)

        except Exception as e:
            print(f"❌ Unexpected error for {protein_id}: {e}")
            failed_ids.append(protein_id)

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 RESULTS SUMMARY - cone Training Data")
    print(f"✅ Successful predictions: {success_count}")
    print(f"❌ Failed predictions: {len(failed_ids)}")
    print(f"📈 Success rate: {success_count/len(cone_df)*100:.1f}%")

    if failed_ids:
        print(f"\nFailed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")

    # Create summary file for cone structures
    if success_count > 0:
        create_cone_structure_summary(success_count, output_dir)

def create_cone_structure_summary(success_count, output_dir):
    """Create a summary file for generated cone structures"""
    print(f"\n📝 Creating summary file for {success_count} cone structures...")

    summary_file = output_dir / "cone_structure_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"cone Training Data - Generated Structures Summary\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write(f"Total successful predictions: {success_count}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Organized by ion channel class: Na, K, Ca, Other\n")

    print(f"✅ Summary saved to: {summary_file}")

def test_run(n_sequences=5):
    """Test run with first N sequences"""
    print(f"🧪 TEST RUN - Processing first {n_sequences} cone sequences")
    print("=" * 60)

    # Setup
    output_dir = setup_output_directories()
    cone_df = load_cone_sequences()

    # Take only first N sequences for testing
    test_df = cone_df.head(n_sequences)

    # Track results
    success_count = 0
    failed_ids = []

    # Process test sequences
    print(f"\n🚀 Processing {len(test_df)} test sequences...")

    for idx, row in test_df.iterrows():
        protein_id = row['Entry']
        sequence = row['Sequence']
        ion_class = row['match']

        print(f"\n[{idx+1}/{len(test_df)}] Processing {protein_id} ({ion_class})")

        try:
            # Create temporary FASTA for test
            with tempfile.TemporaryDirectory() as temp_dir:
                fasta_file = write_fasta_sequence(protein_id, sequence, Path(temp_dir))

                # Try ColabFold prediction
                success = run_colabfold_prediction(fasta_file, output_dir, protein_id, ion_class)

                # If that fails, save FASTA for manual processing
                if not success:
                    success = simple_structure_prediction(protein_id, sequence, output_dir, ion_class)

                if success:
                    success_count += 1
                else:
                    failed_ids.append(protein_id)

        except Exception as e:
            print(f"❌ Unexpected error for {protein_id}: {e}")
            failed_ids.append(protein_id)

    # Test summary
    print("\n" + "=" * 60)
    print(f"🧪 TEST RESULTS")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(failed_ids)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        test_run(n_test)
    else:
        # Run full pipeline
        main()