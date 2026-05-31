import os
import pandas as pd
import random


class Preprocessor:
    def __init__(self):
        self.raw_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw",
                                          "uniprot_petase_raw.csv")
        self.processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def clean_and_format(self):
        print("Loading raw UniProt sequences...")
        try:
            df = pd.read_csv(self.raw_data_path)
        except FileNotFoundError:
            print("ERROR: Could not find uniprot_petase_raw.csv. Run data_loader.py first.")
            return

        print("Cleaning sequences and removing invalid amino acids...")
        # Drop any rows without sequences
        df = df.dropna(subset=['sequence'])

        # Filter out sequences with non-standard amino acids (X, B, Z, J)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        df = df[df['sequence'].apply(lambda seq: all(aa in valid_aa for aa in seq))]

        print("Generating synthetic mutation dataset for DoWhy causal mapping...")
        mutation_records = []

        # For each valid sequence, generate some synthetic variants
        for _, row in df.iterrows():
            base_seq = row['sequence']
            base_length = len(base_seq)

            # Keep the wild-type
            mutation_records.append({
                "protein_name": row['protein_name'],
                "sequence": base_seq,
                "is_mutated": 0,
                "mutated_positions": "None",
                "kcat_km_improvement": 1.0  # Baseline
            })

            # Generate 5 synthetic mutations per sequence to simulate literature data
            for _ in range(5):
                # Pick 1 to 3 random positions to mutate
                num_mutations = random.randint(1, 3)
                positions = random.sample(range(base_length), num_mutations)
                positions.sort()

                # Create a mock kinetic score (some good, some bad)
                mock_kinetics = round(random.uniform(0.1, 5.5), 2)

                mutation_records.append({
                    "protein_name": row['protein_name'],
                    "sequence": base_seq,
                    "is_mutated": 1,
                    "mutated_positions": str(positions),
                    "kcat_km_improvement": mock_kinetics
                })

        final_df = pd.DataFrame(mutation_records)
        output_path = os.path.join(self.processed_dir, "literature_mutations.csv")
        final_df.to_csv(output_path, index=False)

        print(f"\n✅ SUCCESS: Generated {len(final_df)} mutation records.")
        print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    print("--- Starting AETHER Preprocessor ---")
    processor = Preprocessor()
    processor.clean_and_format()