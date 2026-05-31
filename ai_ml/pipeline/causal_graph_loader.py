import os
import json


class CausalGraphLoader:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.positions_path = os.path.join(self.models_dir, "causal_positions.json")

    def load_target_positions(self):
        """Loads the Top 10 causal positions for the Swarm to target."""
        print("Fetching DoWhy Causal Target Positions...")

        if not os.path.exists(self.positions_path):
            print("WARNING: causal_positions.json not found! Using emergency fallback targets.")
            return [10, 20, 30, 40, 50]  # Fallback data just in case

        with open(self.positions_path, "r") as f:
            positions = json.load(f)

        print(f"✅ Successfully loaded {len(positions)} causal targets: {positions}")
        return positions


if __name__ == "__main__":
    print("--- Testing Causal Graph Loader ---")
    loader = CausalGraphLoader()
    targets = loader.load_target_positions()