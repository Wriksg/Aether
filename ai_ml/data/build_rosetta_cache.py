import os
import sys
import json
import time

# --- DYNAMIC PATH RESOLUTION BOILERPLATE ---
# Since this file is in ai_ml/data/, we traverse 3 levels up to find the project root (AETHER).
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_ml.brain.sequence_loader import SequenceLoader

# --- AMINO ACID BIOPHYSICAL PROPERTIES ---
# Standard Kyte-Doolittle Hydrophobicity index
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Average side-chain molecular volume (cubic Angstroms)
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 111.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}

# Isoelectric net charge approximation at physiological pH (7.4)
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def compute_rosetta_penalty(pos: int, aa_old: str, aa_new: str) -> float:
    """
    Computes a realistic biophysical pseudo-Rosetta delta-delta G penalty (kcal/mol).
    - Favorable/neutral mutations receive values close to <= 0.0.
    - Destabilizing mutations (such as packing clashes or charge mismatches)
      receive positive penalties up to +6.0.
    """
    if aa_old == aa_new:
        return 0.0

    # Calculate parameter deltas
    dh = HYDROPHOBICITY[aa_new] - HYDROPHOBICITY[aa_old]
    dv = abs(VOLUME[aa_new] - VOLUME[aa_old])

    # Model pocket specific environments:
    # positions 3, 5, 8 are on an exposed surface loop.
    # positions 12, 18, 25 are inside a hydrophobic active site cleft.
    if pos in [3, 5, 8]:  # Exposed surface loop pocket
        # Exposing large hydrophobic groups creates aggregation risk (unfavorable)
        penalty = 0.4 * (dh if dh > 0 else 0.0) + 0.01 * dv
    else:  # Core/cleft hydrophobic pocket
        # Placing highly polar/charged residues inside a hydrophobic core is highly destabilizing
        penalty = 1.3 * (-dh if dh < 0 else 0.0) + 0.015 * dv
        if abs(CHARGE[aa_new]) > 0:
            penalty += 1.8  # Electrostatic solvation penalty

    # Center and scale to standard Rosetta energies (kcal/mol)
    ddg = round(max(-1.50, min(6.00, penalty - 0.40)), 3)
    return ddg


def build_cache():
    print("==================================================")
    print("      AETHER ROSETTA PHYSICS CACHE GENERATOR       ")
    print("==================================================")

    start_time = time.time()

    # 1. Load biological reference sequences
    loader = SequenceLoader()
    wt_seq, fast_seq = loader.load_sequences()

    # Target positions to pre-calculate
    targets = [3, 5, 8, 12, 18, 25]
    print(f"Pre-calculating mutations for target positions: {targets}")

    cache = {}

    # Add baseline wild-type sequences with 0.0 penalty
    cache[wt_seq] = 0.0
    cache[fast_seq] = 0.0

    # Generate all single-point mutations for both reference sequences
    for base_name, base_seq in [("Wildtype", wt_seq), ("FAST-PETase", fast_seq)]:
        base_list = list(base_seq)
        mutations_generated = 0

        for pos in targets:
            if pos >= len(base_list):
                continue
            aa_old = base_list[pos]
            for aa_new in AMINO_ACIDS:
                if aa_new == aa_old:
                    continue

                # Create mutant sequence
                mutant_list = base_list.copy()
                mutant_list[pos] = aa_new
                mutant_seq = "".join(mutant_list)

                # Calculate pseudo-Rosetta delta penalty
                ddg = compute_rosetta_penalty(pos, aa_old, aa_new)
                cache[mutant_seq] = ddg
                mutations_generated += 1

        print(f"-> Generated {mutations_generated} physics profiles for {base_name} scaffold.")

    # 2. Write the cached profiles to models/rosetta_cache.json
    models_dir = os.path.join(project_root, "ai_ml", "models")
    os.makedirs(models_dir, exist_ok=True)
    cache_path = os.path.join(models_dir, "rosetta_cache.json")

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    execution_time = time.time() - start_time
    file_size_kb = os.path.getsize(cache_path) / 1024.0

    # Calculate simple stats
    penalties = list(cache.values())
    min_penalty = min(penalties)
    max_penalty = max(penalties)
    avg_penalty = sum(penalties) / len(penalties)

    print("\n--------------------------------------------------")
    print("               CACHE BUILD SUMMARY                ")
    print("--------------------------------------------------")
    print(f"Status:             SUCCESS")
    print(f"Target Path:        {cache_path}")
    print(f"Total Cached Keys:  {len(cache)}")
    print(f"File Size:          {file_size_kb:.2f} KB")
    print(f"Execution Time:     {execution_time * 1000.0:.2f} ms")
    print(f"Min Delta Energy:   {min_penalty:.3f} kcal/mol (Stabilizing)")
    print(f"Max Delta Energy:   {max_penalty:.3f} kcal/mol (Destabilizing)")
    print(f"Avg Delta Energy:   {avg_penalty:.3f} kcal/mol")
    print("==================================================")


if __name__ == "__main__":
    build_cache()