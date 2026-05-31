# outer_world_optimizer.py
import random
import time
from typing import List, Dict


class OuterWorldOptimizer:
    def __init__(self, target_positions: List[int], wild_type: str, reservoir_size: int = 4):
        self.target_positions = target_positions
        self.wild_type = wild_type
        self.reservoir_size = reservoir_size

        # The Outer World Reservoir: Stores variants that are sub-optimal but structurally diverse
        # Format: [{"sequence": str, "score": float, "mutated_positions": set}]
        self.outer_world_reservoir: List[Dict] = []

        # Track globally recognized best sequence
        self.global_best_seq = wild_type
        self.global_best_score = self._evaluate_epistatic_landscape(wild_type)

    def _evaluate_epistatic_landscape(self, sequence: str) -> float:
        """
        A simulated non-linear epistatic fitness landscape.
        Mutating position 3 alone is bad (-2.0)
        Mutating position 8 alone is bad (-2.0)
        Mutating BOTH positions 3 and 8 yields a massive synergistic peak (+10.0)
        """
        score = 0.0
        has_mut_3 = (sequence[3] == 'M')
        has_mut_8 = (sequence[8] == 'M')

        # Sign Epistasis Rule
        if has_mut_3 and has_mut_8:
            score += 10.0  # The Synergistic Peak (Bypassed the valley)
        elif has_mut_3 or has_mut_8:
            score -= 2.0  # The Evolutionary Valley of Death (Deleterious intermediate)
        else:
            score = 0.0  # Wild-type baseline

        # Add slight positive noise for other positions to simulate natural drift
        for pos in range(len(sequence)):
            if pos not in [3, 8] and sequence[pos] != self.wild_type[pos]:
                score += 0.1

        return score

    def _update_outer_world_reservoir(self, sequence: str, score: float):
        """
        Incorporate sequences into the Outer World Reservoir based on diversity,
        even if they have low/sub-optimal fitness scores.
        """
        # Identify which causal positions are mutated
        mutated_positions = {pos for pos in self.target_positions if sequence[pos] != self.wild_type[pos]}
        if not mutated_positions:
            return

        # Check if we already have a candidate covering these exact positions
        for candidate in self.outer_world_reservoir:
            if candidate["mutated_positions"] == mutated_positions:
                # If this candidate is slightly better, replace it, otherwise keep current
                if score > candidate["score"]:
                    candidate["sequence"] = sequence
                    candidate["score"] = score
                return

        # If the reservoir is not full, add it directly
        if len(self.outer_world_reservoir) < self.reservoir_size:
            self.outer_world_reservoir.append({
                "sequence": sequence,
                "score": score,
                "mutated_positions": mutated_positions
            })
            print(
                f" [Outer World] Preserved epistatic seed {sequence} (Score: {score:.1f}) targeting positions {list(mutated_positions)}")
        else:
            # If full, replace the candidate with the lowest mutational unique value or lowest score
            self.outer_world_reservoir.sort(key=lambda x: x["score"])
            if score > self.outer_world_reservoir[0]["score"]:
                old = self.outer_world_reservoir[0]["sequence"]
                self.outer_world_reservoir[0] = {
                    "sequence": sequence,
                    "score": score,
                    "mutated_positions": mutated_positions
                }
                print(f" [Outer World] Replaced seed {old} with new epistatic seed {sequence}")

    def crossover(self, parent1: str, parent2: str) -> str:
        """Uniform crossover to combine distinct mutational patterns."""
        child = list(parent1)
        for pos in range(len(parent1)):
            if random.random() < 0.5:
                child[pos] = parent2[pos]
        return "".join(child)

    def mutate(self, sequence: str, force_pos: int = None) -> str:
        """Mutate a sequence. Optionally force a mutation at a specific causal position."""
        seq_list = list(sequence)
        if force_pos is not None and force_pos < len(seq_list):
            seq_list[force_pos] = 'M'
        else:
            # Random mutation at a causal position
            pos = random.choice(self.target_positions)
            seq_list[pos] = 'M' if random.random() < 0.8 else 'A'
        return "".join(seq_list)

    def optimize(self, generations: int = 15, swarm_size: int = 6):
        print(f"Wild-Type Baseline: {self.wild_type} (Score: {self.global_best_score:.1f})")
        print(f"Targeting Causal Positions: {self.target_positions}\n")

        # Initialize swarm particles near wild-type
        particles = [{"current": self.wild_type, "score": self.global_best_score, "stagnation": 0} for _ in
                     range(swarm_size)]

        for gen in range(generations):
            print(f"\n--- Generation {gen + 1}/{generations} ---")

            for i, p in enumerate(particles):
                # Standard mutation exploration
                candidate = self.mutate(p["current"])
                score = self._evaluate_epistatic_landscape(candidate)

                # Update global best if found
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_seq = candidate
                    print(f" *** NEW GLOBAL BEST *** Found variant: {candidate} with Score: {score:.1f}")

                # Maintain the Outer World Reservoir
                # Even if the candidate score is negative (deleterious intermediate), we cache it!
                if score < 0:
                    self._update_outer_world_reservoir(candidate, score)

                # Inner World Greedy Acceptance
                if score > p["score"]:
                    p["current"] = candidate
                    p["score"] = score
                    p["stagnation"] = 0
                else:
                    p["stagnation"] += 1

                    # Replace the old tunneling block (around line 118) with this:

                    # TUNNELING EVENT: If particle is stuck, pull from the Outer World Reservoir
                    if p["stagnation"] >= 3:
                        if len(self.outer_world_reservoir) >= 2:
                            # 1. Multi-Seed Recombination: Cross two distinct seeds from the reservoir
                            seeds = random.sample(self.outer_world_reservoir, 2)
                            tunnel_candidate = self.crossover(seeds[0]["sequence"], seeds[1]["sequence"])
                            print(
                                f" [Tunneling] Stagnation resolved. Recombining Reservoir Seeds: {seeds[0]['sequence']} x {seeds[1]['sequence']} -> Result: {tunnel_candidate}")
                        elif self.outer_world_reservoir:
                            # 2. Seed-Particle Crossover with forced mutation
                            seed = random.choice(self.outer_world_reservoir)
                            crossed = self.crossover(p["current"], seed["sequence"])
                            tunnel_candidate = self.mutate(crossed)  # Introduce the second mutation
                            print(
                                f" [Tunneling] Stagnation resolved. Crossing Particle with Seed + Mutating -> Result: {tunnel_candidate}")
                        else:
                            continue

                        tunnel_score = self._evaluate_epistatic_landscape(tunnel_candidate)

                        if tunnel_score > p["score"]:
                            p["current"] = tunnel_candidate
                            p["score"] = tunnel_score
                            p["stagnation"] = 0

                        if tunnel_score > self.global_best_score:
                            self.global_best_score = tunnel_score
                            self.global_best_seq = tunnel_candidate
                            print(
                                f" *** SYNERGISTIC PEAK DISCOVERED VIA TUNNELING *** Variant: {tunnel_candidate} Score: {tunnel_score:.1f}")

        print("\n================ OPTIMIZATION SUMMARY ================")
        print(f"Final Global Best Sequence: {self.global_best_seq}")
        print(f"Final Global Best Score:   {self.global_best_score:.1f}")
        print(f"Outer World Reservoir Seeds preserved: {[x['sequence'] for x in self.outer_world_reservoir]}")

        if self.global_best_score == 10.0:
            print("\nSUCCESS: The swarm mathematically bypassed the Evolutionary Valley of Death!")
        else:
            print("\nFAILURE: Standard local search rules prevailed.")


if __name__ == "__main__":
    # We target position 3 and 8 (0-indexed indices)
    opt = OuterWorldOptimizer(target_positions=[3, 8], wild_type="AAAAAAAAAA")
    opt.optimize()