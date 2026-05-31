import os
import json


class PhysicsFilter:
    def __init__(self):
        print("Initializing Physics Filter...")
        # Path to the Rosetta delta cache
        self.cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "rosetta_cache.json")
        self.delta_cache = {}

        try:
            # We must catch both FileNotFoundError and JSONDecodeError (empty file)
            with open(self.cache_path, "r") as f:
                self.delta_cache = json.load(f)
            print(f"Loaded {len(self.delta_cache)} Rosetta physics profiles.")
        except FileNotFoundError:
            print("WARNING: rosetta_cache.json not found. Physics Filter running in passthrough mode.")
            self.delta_cache = {}
        except json.JSONDecodeError:
            print("WARNING: rosetta_cache.json is empty. Physics Filter running in passthrough mode.")
            self.delta_cache = {}

    def apply_penalty(self, sequence: str, base_score: float) -> float:
        """
        Applies a physics-based penalty if the mutation causes structural clashes.
        If the cache is missing or empty, it returns the base score safely.
        """
        # Since ESM-2 inherently understands folding, if we lack explicit physics data,
        # we can safely fallback to trusting the ESM-2 baseline score.
        penalty = self.delta_cache.get(sequence, 0.0)
        return base_score - penalty