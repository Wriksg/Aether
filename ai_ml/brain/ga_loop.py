import random
from typing import List, Tuple

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class GeneticAlgorithm:
    def __init__(self, config, target_positions: List[int]):
        self.config = config
        self.target_positions = target_positions

    def initialize_population(self, wild_type: str) -> List[str]:
        population = [wild_type]
        for _ in range(self.config.population_size - 1):
            population.append(self.mutate(wild_type, force=True))
        return population

    def mutate(self, sequence: str, force: bool = False) -> str:
        if not force and random.random() > self.config.mutation_rate:
            return sequence
            
        seq_list = list(sequence)
        # Select 1-based biological position
        bio_pos = random.choice(self.target_positions)
        # Convert to 0-based array index
        idx = bio_pos - 1 
        
        new_aa = random.choice(AMINO_ACIDS)
        while new_aa == seq_list[idx]:
            new_aa = random.choice(AMINO_ACIDS)
            
        seq_list[idx] = new_aa
        return "".join(seq_list)

    def crossover(self, parent1: str, parent2: str) -> str:
        if random.random() > self.config.crossover_rate:
            return parent1 if random.random() > 0.5 else parent2
            
        child_list = list(parent1)
        for bio_pos in self.target_positions:
            idx = bio_pos - 1  # Convert to 0-based array index
            if random.random() > 0.5:
                child_list[idx] = parent2[idx]
                
        return "".join(child_list)

    def crossover(self, parent1: str, parent2: str) -> str:
        if random.random() > self.config.crossover_rate:
            return parent1 if random.random() > 0.5 else parent2

        child_list = list(parent1)
        for pos in self.target_positions:
            if random.random() > 0.5:
                child_list[pos] = parent2[pos]

        return "".join(child_list)

    def tournament_selection(self, pop_with_scores: List[Tuple[str, float]], k: int = 3) -> str:
        tournament = random.sample(pop_with_scores, k)
        return max(tournament, key=lambda x: x[1])[0]

    def apply_diversity_penalty(self, pop_with_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        seen = set()
        penalized = []
        for seq, score in pop_with_scores:
            if seq in seen:
                score -= 5.0
            else:
                seen.add(seq)
            penalized.append((seq, score))
        return penalized

    def evolve_one_generation(
        self,
        current_population: List[str],
        esm2_scorer,
        physics_filter
    ) -> List[str]:
        # Score
        pop_with_scores = []
        for seq in current_population:
            raw_score = esm2_scorer.get_fitness(seq)
            final_score = physics_filter.apply_penalty(seq, raw_score)
            pop_with_scores.append((seq, final_score))

        # Diversity penalty
        pop_with_scores = self.apply_diversity_penalty(pop_with_scores)

        # Elitism
        new_population = [max(pop_with_scores, key=lambda x: x[1])[0]]

        # Fill generation
        while len(new_population) < self.config.population_size:
            p1 = self.tournament_selection(pop_with_scores)
            p2 = self.tournament_selection(pop_with_scores)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        return new_population