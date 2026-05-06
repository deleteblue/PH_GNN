"""
Deduplicate taxa FASTA files: keep one representative per unique sequence.
Writes:
    taxa_fastas/deduplication_report.csv  - all groups, kept/dropped, discrepancies
    taxa_fastas/Cone_Snail.fasta          - deduplicated (overwritten)
    taxa_fastas/*.fasta                   - deduplicated (overwritten)
"""

import os
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "taxa_fastas")

TAXA_FILES = [
    "Cone_Snail.fasta",
    "Scorpion.fasta",
    "Sea_Anemone.fasta",
    "Snake.fasta",
    "Spider.fasta",
]

DATASET_PRIORITY = {"toxinCombined": 0, "cone": 1, "nonIonToxin": 2}


def build_lookup():
    df_non = pd.read_excel(os.path.join(BASE, "nonIonToxinCombined.xlsx"))
    df_tox = pd.read_excel(os.path.join(BASE, "toxinCombined_Finalv3.xlsx"))
    df_cone = pd.read_excel(os.path.join(BASE, "cone_trainingData.xlsx"))

    lookup = {}
    for _, r in df_non.iterrows():
        lookup[str(r["entry"])] = {"dataset": "nonIonToxin", "ion_class": "nonIonChannel", "organism": str(r["organism"])}
    for _, r in df_tox.iterrows():
        lookup[str(r["entry"])] = {"dataset": "toxinCombined", "ion_class": str(r["ionChannelClass"]), "organism": str(r["organism"])}
    for _, r in df_cone.iterrows():
        pid = str(r["Entry"])
        if pid not in lookup:
            lookup[pid] = {"dataset": "cone", "ion_class": str(r["match"]), "organism": str(r.get("Organism", ""))}
    return lookup


def parse_fasta(filepath):
    entries = []
    header, seq_lines = None, []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    entries.append((header, "".join(seq_lines)))
                header, seq_lines = line, []
            else:
                seq_lines.append(line)
    if header:
        entries.append((header, "".join(seq_lines)))
    return entries


def write_fasta(filepath, entries):
    with open(filepath, "w") as f:
        for header, seq in entries:
            f.write(f"{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


def pick_representative(ids, lookup):
    """Pick best ID: prefer toxinCombined > cone > nonIonToxin, then shortest ID."""
    def priority(pid):
        ds = lookup.get(pid, {}).get("dataset", "nonIonToxin")
        return (DATASET_PRIORITY.get(ds, 99), len(pid), pid)
    return sorted(ids, key=priority)[0]


def main():
    lookup = build_lookup()
    report_rows = []
    total_before, total_after = 0, 0

    for fname in TAXA_FILES:
        taxa = fname.replace(".fasta", "").replace("_", " ")
        path = os.path.join(OUT_DIR, fname)
        entries = parse_fasta(path)
        total_before += len(entries)

        # Group by sequence
        seq_to_ids = defaultdict(list)
        id_to_entry = {}
        for header, seq in entries:
            pid = header.split()[0][1:]
            seq_to_ids[seq].append(pid)
            id_to_entry[pid] = (header, seq)

        kept_entries = []
        for seq, ids in seq_to_ids.items():
            if len(ids) == 1:
                kept_entries.append(id_to_entry[ids[0]])
                continue

            # Check for ion class discrepancy within group
            classes = list({lookup.get(pid, {}).get("ion_class", "unknown") for pid in ids})
            has_discrepancy = len(classes) > 1
            discrepancy_note = f"ion_class conflict: {classes}" if has_discrepancy else ""

            kept_id = pick_representative(ids, lookup)
            dropped_ids = [pid for pid in ids if pid != kept_id]
            kept_entries.append(id_to_entry[kept_id])

            kept_info = lookup.get(kept_id, {})
            for did in dropped_ids:
                d_info = lookup.get(did, {})
                report_rows.append({
                    "taxa": taxa,
                    "action": "dropped",
                    "protein_id": did,
                    "kept_instead": kept_id,
                    "group_ids": ",".join(ids),
                    "seq_length": len(seq),
                    "dropped_dataset": d_info.get("dataset", "unknown"),
                    "dropped_ion_class": d_info.get("ion_class", "unknown"),
                    "kept_dataset": kept_info.get("dataset", "unknown"),
                    "kept_ion_class": kept_info.get("ion_class", "unknown"),
                    "ion_class_discrepancy": discrepancy_note,
                })

            # Also log the kept entry
            report_rows.append({
                "taxa": taxa,
                "action": "kept",
                "protein_id": kept_id,
                "kept_instead": "",
                "group_ids": ",".join(ids),
                "seq_length": len(seq),
                "dropped_dataset": "",
                "dropped_ion_class": "",
                "kept_dataset": kept_info.get("dataset", "unknown"),
                "kept_ion_class": kept_info.get("ion_class", "unknown"),
                "ion_class_discrepancy": discrepancy_note,
            })

        write_fasta(path, kept_entries)
        total_after += len(kept_entries)
        removed = len(entries) - len(kept_entries)
        print(f"  {taxa:12s}: {len(entries):4d} -> {len(kept_entries):4d}  (removed {removed})")

    # Write report
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(OUT_DIR, "deduplication_report.csv")
    report_df.to_csv(report_path, index=False)

    print(f"\nTotal: {total_before} -> {total_after} sequences ({total_before - total_after} removed)")
    print(f"Report: deduplication_report.csv ({len(report_df)} rows)")

    # Flag discrepancies
    disc = report_df[report_df["ion_class_discrepancy"] != ""]
    if not disc.empty:
        print(f"\n⚠️  ION CLASS DISCREPANCIES ({len(disc['group_ids'].unique())} group(s)):")
        for _, row in disc.drop_duplicates("group_ids").iterrows():
            print(f"  Group: {row['group_ids']}")
            print(f"  {row['ion_class_discrepancy']}")
            print(f"  -> Kept: {row['kept_instead'] or row['protein_id']} ({row['kept_ion_class']} / {row['kept_dataset']})")


if __name__ == "__main__":
    print("Deduplicating taxa FASTA files...")
    main()
