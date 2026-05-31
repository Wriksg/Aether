from ai_ml.brain.ga_loop import GeneticAlgorithm
from ai_ml.brain.physics_filter import PhysicsFilter


# Mock imports for testing - replace with your actual class paths if different
class MockConfig:
    population_size = 5
    mutation_rate = 0.1
    crossover_rate = 0.7


class MockScorer:
    def get_fitness(self, seq):
        return -0.5  # Mock log-likelihood score


if __name__ == "__main__":
    print("--- Testing GA Loop Integration ---")

    # Setup dependencies
    config = MockConfig()
    targets = [3, 5, 8]  # Causal positions
    ga = GeneticAlgorithm(config, targets)
    scorer = MockScorer()
    physics = PhysicsFilter()  # Should load empty/fallback

    # Test initialization
    wild_type = "MSTAGGERLFQGVNESLAAQFNPAVANCYHSTGDLGLDHFESPASYSVSSNT"
    population = ga.initialize_population(wild_type)

    # Test one generation evolution
    new_pop = ga.evolve_one_generation(population, scorer, physics)

    print(f"✅ GA Loop successfully evolved a population of {len(new_pop)} candidates.")
    print(f"Sample candidate: {new_pop[0][:20]}...")