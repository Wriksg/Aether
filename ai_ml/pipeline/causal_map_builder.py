import os
import json
import pandas as pd
from collections import defaultdict


class CausalMapBuilder:
    def __init__(self):
        self.processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def build_causal_map(self):
        print("Loading synthetic literature mutations...")
        df_path = os.path.join(self.processed_dir, "literature_mutations.csv")
        try:
            df = pd.read_csv(df_path)
        except FileNotFoundError:
            print("ERROR: literature_mutations.csv not found. Run preprocessor.py first.")
            return

        print("Initializing Causal Inference Model...")
        position_effects = defaultdict(list)

        for _, row in df.iterrows():
            # FIX: Pandas turns "None" into a float NaN. We must force it back to a string!
            mut_pos = str(row['mutated_positions'])
            if mut_pos != "None" and mut_pos != "nan":
                # Parse the string representation of the list
                positions = eval(mut_pos)
                improvement = float(row['kcat_km_improvement'])

                for pos in positions:
                    position_effects[pos].append(improvement)

        print("Calculating Average Treatment Effects (ATE)...")
        causal_results = {}
        for pos, effects in position_effects.items():
            # Calculate ATE for each amino acid position
            ate = sum(effects) / len(effects)
            causal_results[pos] = round(ate, 3)

        # Sort by highest causal impact
        sorted_positions = sorted(causal_results.items(), key=lambda x: x[1], reverse=True)

        # Extract the top 10 most causally linked positions for the Swarm to target
        target_positions = [pos for pos, score in sorted_positions[:10]]

        # Save the full causal graph data
        causal_map_path = os.path.join(self.model_dir, "causal_map.json")
        with open(causal_map_path, "w") as f:
            json.dump(causal_results, f, indent=4)

        # Save the specific target positions for inference_engine.py
        causal_positions_path = os.path.join(self.model_dir, "causal_positions.json")
        with open(causal_positions_path, "w") as f:
            json.dump(target_positions, f)

        print(f"\n✅ SUCCESS: Built Causal Map with {len(causal_results)} nodes.")
        print(f"🎯 Identified Top {len(target_positions)} High-Impact Positions: {target_positions}")
        print(f"📁 Saved target positions to: {causal_positions_path}")


if __name__ == "__main__":
    print("--- Starting AETHER Causal Map Builder ---")
    builder = CausalMapBuilder()
    builder.build_causal_map()