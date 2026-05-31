import random
from typing import List, Dict

from ai_ml.brain.ga_loop import GeneticAlgorithm
from ai_ml.brain.physics_filter import PhysicsFilter

# Import the scorer safely, with fallback support for local unit testing
try:
    from ai_ml.brain.esm2_scorer import ESM2Scorer
except ImportError:
    ESM2Scorer = None


class PSOOptimizer:
    def __init__(self, config, target_positions: List[int], wild_type: str = None, scorer=None, physics_filter=None):
        """
        Swarm-GA Hybrid Optimizer with a Parallel Epistatic (Outer World) Reservoir
        and local stagnation-triggered quantum-inspired tunneling crossovers.
        """
        self.config = config
        self.target_positions = target_positions
        self.wild_type = wild_type

        self.ga_engine = GeneticAlgorithm(config, target_positions)

        # Share resource pointers to prevent memory fragmentation on CPU
        self.physics_filter = physics_filter if physics_filter is not None else PhysicsFilter()
        self.scorer = scorer

        self.global_best_seq = None
        self.global_best_score = -float('inf')
        self.generations_completed = 0

        # High-performance cache to avoid redundant neural network evaluations
        self.score_cache = {}

        # --- PARALLEL EPISTATIC RESERVOIR (OUTER WORLD BUFFER) ---
        self.outer_world_reservoir: List[Dict] = []
        self.reservoir_size = getattr(config, "reservoir_size", 4)

    def _update_outer_world_reservoir(self, sequence: str, score: float):
        """
        Preserves structurally diverse sequences in the Outer World Reservoir
        based on causal positions, even if their scores are sub-optimal.
        """
        if not self.wild_type:
            return

        # Track mutations strictly at target causal positions
        mutated_positions = {
            pos for pos in self.target_positions
            if pos < len(sequence) and pos < len(self.wild_type) and sequence[pos] != self.wild_type[pos]
        }
        if not mutated_positions:
            return

        # If a candidate already targets these exact positions, keep the better-scoring one
        for candidate in self.outer_world_reservoir:
            if candidate["mutated_positions"] == mutated_positions:
                if score > candidate["score"]:
                    candidate["sequence"] = sequence
                    candidate["score"] = score
                return

        # If reservoir is not full, add directly; otherwise, replace the lowest-scoring seed
        if len(self.outer_world_reservoir) < self.reservoir_size:
            self.outer_world_reservoir.append({
                "sequence": sequence,
                "score": score,
                "mutated_positions": mutated_positions
            })
            print(f" [Outer World] Preserved epistatic seed {sequence[:10]}... (Score: {score:.4f}) targeting causal sites {list(mutated_positions)}")
        else:
            self.outer_world_reservoir.sort(key=lambda x: x["score"])
            if score > self.outer_world_reservoir[0]["score"]:
                self.outer_world_reservoir[0] = {
                    "sequence": sequence,
                    "score": score,
                    "mutated_positions": mutated_positions
                }

    def _score(self, sequence: str) -> float:
        """Calculates fitness, skipping neural evaluations via cached records when possible."""
        if sequence in self.score_cache:
            return self.score_cache[sequence]

        # Calculate delta score against the wild-type baseline using the shared scorer
        if self.scorer:
            raw = self.scorer.get_fitness(sequence, wild_type=self.wild_type)
        else:
            raw = -2.0 + (random.random() * 0.5)

        final_score = self.physics_filter.apply_penalty(sequence, raw)

        # Cache score for future generations
        self.score_cache[sequence] = final_score
        return final_score

    def optimize(self, wild_type: str) -> List[Dict]:
        print(f"Initializing swarm of {self.config.swarm_size} particles with Outer World Reservoir...")

        if wild_type and not self.wild_type:
            self.wild_type = wild_type

        initial_population = self.ga_engine.initialize_population(wild_type)
        swarm_seqs = initial_population[:self.config.swarm_size]

        # Initialize particles with local stagnation trackers
        particles = []
        for seq in swarm_seqs:
            score = self._score(seq)
            particles.append({
                "current": seq,
                "pbest": seq,
                "pbest_score": score,
                "stagnation_count": 0  # Tracks local individual stagnation
            })
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_seq = seq

        generations_without_improvement = 0

        for gen in range(self.config.generations):
            improvement_this_gen = False

            for i, particle in enumerate(particles):
                r = random.random()
                candidate = None
                tunneling_occurred = False

                # --- STAGNATION-TRIGGERED TUNNELING CROSSOVER ---
                particle_patience = getattr(self.config, "particle_patience", 3)
                if particle.get("stagnation_count", 0) >= particle_patience:
                    if len(self.outer_world_reservoir) >= 2:
                        # Recombine two distinct sub-optimal seeds from the reservoir
                        seeds = random.sample(self.outer_world_reservoir, 2)
                        candidate = self.ga_engine.crossover(seeds[0]["sequence"], seeds[1]["sequence"])
                        print(f" [Tunneling] Particle {i} stagnation resolved. Recombining Reservoir Seeds: {seeds[0]['sequence'][:10]}... x {seeds[1]['sequence'][:10]}...")
                        tunneling_occurred = True
                    elif self.outer_world_reservoir:
                        # Recombine particle's current sequence with single available seed and mutate
                        seed = random.choice(self.outer_world_reservoir)
                        crossed = self.ga_engine.crossover(particle["current"], seed["sequence"])
                        candidate = self.ga_engine.mutate(crossed, force=True)
                        print(f" [Tunneling] Particle {i} stagnation resolved. Crossing Particle with Seed + Mutating.")
                        tunneling_occurred = True

                # Fallback to standard optimization strategies if tunneling did not occur
                if not candidate:
                    # Strategy 1: Social Pull + Forced Mutation to escape local optima
                    if r < 0.33 and self.global_best_seq:
                        candidate = self.ga_engine.crossover(
                            particle["current"],
                            self.global_best_seq
                        )
                        candidate = self.ga_engine.mutate(candidate, force=True)

                    # Strategy 2: Cognitive Pull (Force mutate from personal best)
                    elif r < 0.66:
                        candidate = self.ga_engine.mutate(particle["pbest"], force=True)

                    # Strategy 3: Exploration Burst (Mutate fresh from wild-type)
                    else:
                        candidate = self.ga_engine.mutate(self.wild_type, force=True)

                new_score = self._score(candidate)

                # Maintain the Outer World Reservoir with diverse structural variations
                if new_score < self.global_best_score:
                    self._update_outer_world_reservoir(candidate, new_score)

                # Update personal best if improved
                if new_score > particle["pbest_score"]:
                    particle["current"] = candidate
                    particle["pbest"] = candidate
                    particle["pbest_score"] = new_score
                    particle["stagnation_count"] = 0  # Reset stagnation count upon progress
                    improvement_this_gen = True

                    # Update global best
                    if new_score > self.global_best_score:
                        self.global_best_score = new_score
                        self.global_best_seq = candidate
                        if tunneling_occurred:
                            print(f" *** SYNERGISTIC PEAK DISCOVERED VIA TUNNELING *** Variant: {candidate[:10]}... Score: {new_score:.6f}")
                        else:
                            print(f"  Gen {gen + 1}: New global best — score {new_score:.6f}")
                else:
                    particle["stagnation_count"] = particle.get("stagnation_count", 0) + 1

            # Early stopping check using `patience` from ga_config
            if improvement_this_gen:
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            self.generations_completed = gen + 1

            if generations_without_improvement >= self.config.patience:
                print(f"Converged at generation {gen + 1} after "
                      f"{self.config.patience} stagnant generations. Early stopping triggered.")
                break  # Stops loop when progress halts across generations

            print(f"Generation {gen + 1}/{self.config.generations} complete. "
                  f"Global best score: {self.global_best_score:.6f}")

        # Build final swarm from unique personal bests
        seen = set()
        final_swarm = []

        for p in particles:
            if p["pbest"] not in seen:
                seen.add(p["pbest"])
                final_swarm.append({"sequence": p["pbest"], "score": p["pbest_score"]})

        # Padding mechanism: Fill to 10 if needed by force-mutating from global best
        attempts = 0
        while len(final_swarm) < 10 and attempts < 200:
            candidate = self.ga_engine.mutate(self.global_best_seq, force=True)
            if candidate not in seen:
                score = self._score(candidate)
                seen.add(candidate)
                final_swarm.append({"sequence": candidate, "score": score})
            attempts += 1

        # Sort the final dictionary array by score (highest first)
        final_swarm.sort(key=lambda x: x["score"], reverse=True)

        print(f"\nOptimization complete. {len(final_swarm)} unique candidates found.")
        print(f"Global best score: {self.global_best_score:.6f}")

        return final_swarm