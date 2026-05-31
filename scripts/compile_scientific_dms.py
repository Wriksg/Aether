# scripts/compile_scientific_dms.py
import os
import sys
import re
import pandas as pd
import numpy as np

# --- DYNAMIC PATH RESOLUTION ---
# Traverse 2 levels up to reach project root (AETHER)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Target biological positions (FAST-PETase and active site hotspots)
CAUSAL_SITES = [121, 159, 186, 206, 224, 233, 237, 280]


def parse_mutations(mut_string: str) -> list[int]:
    """
    Standardizes and parses mutational delimiters across different database formats:
    Handles colons ('S121E:D186H'), semicolons ('S121E; D186H'), or single mutations.
    """
    if pd.isna(mut_string) or str(mut_string).lower() in ["wt", "wildtype", "wild-type", "none"]:
        return []
    
    # Split across colons, semicolons, commas, or whitespace
    muts = re.split(r'[:;, \s]+', str(mut_string))
    positions = []
    
    for m in muts:
        nums = re.findall(r'\d+', m)
        if nums:
            positions.append(int(nums[0]))
            
    return positions


def compile_dms_data():
    raw_dir = os.path.join(project_root, "ai_ml", "data", "raw")
    output_path = os.path.join(raw_dir, "mutation_dataset.csv")
    
    all_variants = []

    # --- PARSING SOURCE B: PROTEINGYM DMS ASSAY ---
    proteingym_file = os.path.join(raw_dir, "proteingym_dms.csv")
    if os.path.exists(proteingym_file):
        print(f"Parsing ProteinGym DMS assay: {proteingym_file}")
        df = pd.read_csv(proteingym_file)
        
        # Verify columns exist
        mut_col = "mutant" if "mutant" in df.columns else [c for c in df.columns if "mut" in c.lower()][0]
        score_col = "DMS_score" if "DMS_score" in df.columns else [c for c in df.columns if "score" in c.lower() or "fitness" in c.lower()][0]
        
        for _, row in df.iterrows():
            pos_list = parse_mutations(row[mut_col])
            record = {f"pos_{p}": (1 if p in pos_list else 0) for p in CAUSAL_SITES}
            record["activity_score"] = float(row[score_col])
            record["source"] = "ProteinGym"
            all_variants.append(record)
    else:
        print(f"\n[!] Error: 'proteingym_dms.csv' not found in: {raw_dir}")
        return

    # Convert to DataFrame
    compiled_df = pd.DataFrame(all_variants)
    
    # Simulate/Add experimental expression efficiency as our causal confounder
    np.random.seed(42)
    compiled_df["expression_efficiency"] = 1.0 + np.random.normal(0, 0.25, len(compiled_df))
    compiled_df["expression_efficiency"] += 0.85
    
    # Re-order columns
    cols = [f"pos_{p}" for p in CAUSAL_SITES] + ["expression_efficiency", "activity_score"]
    compiled_df = compiled_df[cols]
    
    # Save output
    compiled_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Successfully compiled {len(compiled_df)} real-world mutational variants into '{output_path}'")


if __name__ == "__main__":
    compile_dms_data()