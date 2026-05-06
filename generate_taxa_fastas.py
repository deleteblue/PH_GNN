"""
Generate one combined FASTA file per taxa from batch1-4 protein FASTA files.
Sources: nonIonToxinCombined, toxinCombined_Finalv3, cone_trainingData

Output:
    taxa_fastas/Snake.fasta
    taxa_fastas/Cone_Snail.fasta
    taxa_fastas/Sea_Anemone.fasta
    taxa_fastas/Spider.fasta
    taxa_fastas/Scorpion.fasta
    taxa_fastas/taxa_metadata.csv
"""

import os
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
BATCH_DIRS = [os.path.join(BASE, f"batch{i}") for i in range(1, 5)]
OUT_DIR = os.path.join(BASE, "taxa_fastas")
os.makedirs(OUT_DIR, exist_ok=True)


def build_taxa_lookup():
    """Build protein_id -> (taxa, organism, dataset) from all three Excel files."""
    lookup = {}

    df_non = pd.read_excel(os.path.join(BASE, "nonIonToxinCombined.xlsx"))
    for _, row in df_non.iterrows():
        lookup[str(row["entry"])] = {
            "taxa": row["commonName"],
            "organism": row["organism"],
            "dataset": "nonIonToxin",
            "protein_name": row.get("proteinName", ""),
        }

    df_tox = pd.read_excel(os.path.join(BASE, "toxinCombined_Finalv3.xlsx"))
    for _, row in df_tox.iterrows():
        lookup[str(row["entry"])] = {
            "taxa": row["commonName"],
            "organism": row["organism"],
            "dataset": "toxinCombined",
            "protein_name": row.get("proteinNames", ""),
        }

    df_cone = pd.read_excel(os.path.join(BASE, "cone_trainingData.xlsx"))
    for _, row in df_cone.iterrows():
        pid = str(row["Entry"])
        if pid not in lookup:
            lookup[pid] = {
                "taxa": "Cone Snail",
                "organism": row.get("Organism", ""),
                "dataset": "cone",
                "protein_name": row.get("Protein names", ""),
            }

    return lookup


def read_fasta(filepath):
    """Read a single-sequence FASTA file, return (header, sequence)."""
    header, seq_lines = "", []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                header = line
            else:
                seq_lines.append(line)
    return header, "".join(seq_lines)


def main():
    print("Building taxa lookup from Excel files...")
    lookup = build_taxa_lookup()
    print(f"  {len(lookup)} proteins mapped to taxa")

    # Collect all FASTA files from batches
    fasta_files = {}
    for batch_dir in BATCH_DIRS:
        batch_name = os.path.basename(batch_dir)
        for fname in os.listdir(batch_dir):
            if not fname.endswith(".fasta"):
                continue
            pid = fname.replace(".fasta", "")
            fasta_files[pid] = (os.path.join(batch_dir, fname), batch_name)

    print(f"  {len(fasta_files)} FASTA files found across batch1-4")

    # Group by taxa
    taxa_sequences = defaultdict(list)  # taxa -> list of (pid, header, seq, batch, info)
    unmatched = []

    for pid, (fpath, batch) in fasta_files.items():
        info = lookup.get(pid)
        if info is None:
            unmatched.append(pid)
            continue
        header, seq = read_fasta(fpath)
        if not seq:
            continue
        taxa_sequences[info["taxa"]].append({
            "protein_id": pid,
            "header": header,
            "sequence": seq,
            "batch": batch,
            "taxa": info["taxa"],
            "organism": info["organism"],
            "dataset": info["dataset"],
            "protein_name": info["protein_name"],
            "sequence_length": len(seq),
        })

    if unmatched:
        print(f"  Warning: {len(unmatched)} proteins not in any Excel file: {unmatched[:5]}")

    # Write one FASTA per taxa + collect metadata
    metadata_rows = []
    for taxa, entries in sorted(taxa_sequences.items()):
        safe_name = taxa.replace(" ", "_")
        out_fasta = os.path.join(OUT_DIR, f"{safe_name}.fasta")
        with open(out_fasta, "w") as f:
            for entry in entries:
                f.write(f"{entry['header']}\n")
                # Write sequence in 60-char lines
                seq = entry["sequence"]
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + "\n")
        print(f"  Wrote {len(entries):4d} sequences -> {safe_name}.fasta")
        metadata_rows.extend(entries)

    # Write metadata CSV
    meta_df = pd.DataFrame(metadata_rows)[
        ["protein_id", "taxa", "organism", "dataset", "batch",
         "sequence_length", "protein_name"]
    ]
    meta_path = os.path.join(OUT_DIR, "taxa_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"\nMetadata: {len(meta_df)} rows -> taxa_metadata.csv")

    # Summary
    print("\nSummary:")
    print(meta_df.groupby(["taxa", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
