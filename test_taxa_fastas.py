"""
Tests for generate_taxa_fastas.py output.
Checks: counts, taxa assignment, no duplicates, sequence integrity, metadata consistency.
"""

import os
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "taxa_fastas")
BATCH_DIRS = [os.path.join(BASE, f"batch{i}") for i in range(1, 5)]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}{': ' + detail if detail else ''}")
        failures.append(name)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_fasta(filepath):
    """Return list of (protein_id, sequence) from a multi-sequence FASTA."""
    entries = []
    header, seq_lines = None, []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    entries.append((header.split()[0][1:], "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
    if header:
        entries.append((header.split()[0][1:], "".join(seq_lines)))
    return entries


def read_single_fasta_seq(filepath):
    """Return sequence string from a single-protein FASTA file."""
    seq_lines = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith(">"):
                seq_lines.append(line)
    return "".join(seq_lines)


# ── Load outputs ───────────────────────────────────────────────────────────────

print("Loading taxa FASTA files and metadata...")
taxa_files = {
    "Cone Snail":  "Cone_Snail.fasta",
    "Scorpion":    "Scorpion.fasta",
    "Sea Anemone": "Sea_Anemone.fasta",
    "Snake":       "Snake.fasta",
    "Spider":      "Spider.fasta",
}

taxa_data = {}
for taxa, fname in taxa_files.items():
    path = os.path.join(OUT_DIR, fname)
    taxa_data[taxa] = parse_fasta(path)

meta = pd.read_csv(os.path.join(OUT_DIR, "taxa_metadata.csv"))
print(f"  Metadata: {len(meta)} rows\n")


# ── 1. File existence ─────────────────────────────────────────────────────────
print("1. File existence")
for taxa, fname in taxa_files.items():
    check(f"{fname} exists", os.path.exists(os.path.join(OUT_DIR, fname)))
check("taxa_metadata.csv exists", os.path.exists(os.path.join(OUT_DIR, "taxa_metadata.csv")))


# ── 2. Total count ────────────────────────────────────────────────────────────
print("\n2. Total sequence count")
total_in_fasta = sum(len(v) for v in taxa_data.values())
batch_total = sum(
    len([f for f in os.listdir(d) if f.endswith(".fasta")])
    for d in BATCH_DIRS
)
check(f"FASTA total ({total_in_fasta}) == batch total ({batch_total})",
      total_in_fasta == batch_total)
check(f"Metadata rows ({len(meta)}) == batch total ({batch_total})",
      len(meta) == batch_total)


# ── 3. Per-taxa counts match metadata ─────────────────────────────────────────
print("\n3. Per-taxa counts match metadata")
for taxa, entries in taxa_data.items():
    meta_count = len(meta[meta["taxa"] == taxa])
    check(f"{taxa}: FASTA ({len(entries)}) == metadata ({meta_count})",
          len(entries) == meta_count)


# ── 4. No duplicate protein IDs ───────────────────────────────────────────────
print("\n4. No duplicate protein IDs")
all_ids = [pid for entries in taxa_data.values() for pid, _ in entries]
unique_ids = set(all_ids)
check(f"All {len(all_ids)} IDs unique", len(all_ids) == len(unique_ids),
      f"{len(all_ids) - len(unique_ids)} duplicates found")
check(f"Metadata protein_id unique", meta["protein_id"].nunique() == len(meta))


# ── 5. No empty sequences ─────────────────────────────────────────────────────
print("\n5. No empty sequences")
for taxa, entries in taxa_data.items():
    empty = [(pid, seq) for pid, seq in entries if not seq.strip()]
    check(f"{taxa}: no empty sequences", len(empty) == 0,
          f"{len(empty)} empty")


# ── 6. Taxa assignment spot-check vs Excel ────────────────────────────────────
print("\n6. Taxa assignment vs Excel source")
df_non = pd.read_excel(os.path.join(BASE, "nonIonToxinCombined.xlsx"))
df_tox = pd.read_excel(os.path.join(BASE, "toxinCombined_Finalv3.xlsx"))
df_cone = pd.read_excel(os.path.join(BASE, "cone_trainingData.xlsx"))

# Build ground-truth taxa lookup — same priority order as the generator:
# nonIonToxin first, toxinCombined second, cone last (only if not already mapped)
gt = {}
for _, r in df_non.iterrows():
    gt[str(r["entry"])] = r["commonName"]
for _, r in df_tox.iterrows():
    gt[str(r["entry"])] = r["commonName"]
for _, r in df_cone.iterrows():
    pid = str(r["Entry"])
    if pid not in gt:
        gt[pid] = "Cone Snail"

# Check every protein in metadata
wrong = []
for _, row in meta.iterrows():
    pid = row["protein_id"]
    expected = gt.get(pid)
    if expected and row["taxa"] != expected:
        wrong.append((pid, expected, row["taxa"]))

check(f"All {len(meta)} proteins assigned correct taxa",
      len(wrong) == 0,
      f"{len(wrong)} wrong: {wrong[:3]}")


# ── 7. cone proteins all in Cone Snail ───────────────────────────────────────
print("\n7. cone proteins all assigned Cone Snail")
cone_in_meta = meta[meta["dataset"] == "cone"]
all_cone = (cone_in_meta["taxa"] == "Cone Snail").all()
check(f"All {len(cone_in_meta)} cone proteins -> Cone Snail", all_cone)


# ── 8. Sequence integrity (random sample vs source FASTA) ─────────────────────
print("\n8. Sequence integrity (20-protein random sample vs source batch files)")

# Map protein_id -> source fasta path
source_map = {}
for d in BATCH_DIRS:
    for fname in os.listdir(d):
        if fname.endswith(".fasta"):
            source_map[fname.replace(".fasta", "")] = os.path.join(d, fname)

# Build output lookup: protein_id -> sequence
output_seq = {pid: seq for entries in taxa_data.values() for pid, seq in entries}

sample = meta.sample(n=min(20, len(meta)), random_state=42)
mismatches = []
for _, row in sample.iterrows():
    pid = row["protein_id"]
    source_seq = read_single_fasta_seq(source_map[pid])
    out_seq = output_seq.get(pid, "")
    if source_seq != out_seq:
        mismatches.append(pid)

check(f"20-protein sample sequences match source files",
      len(mismatches) == 0,
      f"Mismatches: {mismatches}")


# ── 9. FASTA format validity ──────────────────────────────────────────────────
print("\n9. FASTA format validity")
for taxa, fname in taxa_files.items():
    path = os.path.join(OUT_DIR, fname)
    bad_headers = []
    with open(path) as f:
        for line in f:
            if line.startswith(">") and len(line.strip()) < 2:
                bad_headers.append(line.strip())
    check(f"{taxa}: all headers non-empty", len(bad_headers) == 0)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if not failures:
    print(f"All tests passed! ({total_in_fasta} sequences across 5 taxa files)")
else:
    print(f"{len(failures)} test(s) FAILED: {failures}")
