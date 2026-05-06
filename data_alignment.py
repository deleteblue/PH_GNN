#!/usr/bin/env python3
import pandas as pd
import glob
import re

# Get training IDs from cone_trainingData.xlsx
df = pd.read_excel('cone_trainingData.xlsx')
training_ids = set(df.iloc[:, 0].astype(str).str.strip())  # Assuming IDs in first column

# Get structure IDs from both folders
structure_ids = set()
for filepath in glob.glob('conotoxin_structures/*.pdb') + glob.glob('conotoxin_structures_2/*.pdb'):
    filename = re.match(r'([A-Z0-9]+)_ranked_\d+\.pdb', filepath.split('/')[-1])
    if filename:
        structure_ids.add(filename.group(1))

# Find mutual IDs
mutual_ids = training_ids.intersection(structure_ids)

# Save mutual IDs if any exist
if mutual_ids:
    with open('mutual_ids.txt', 'w') as f:
        for protein_id in sorted(mutual_ids):
            f.write(f"{protein_id}\n")
    print(f"Found {len(mutual_ids)} mutual IDs saved to mutual_ids.txt")
else:
    print("No mutual IDs found")