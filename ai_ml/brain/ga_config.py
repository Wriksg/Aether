from dataclasses import dataclass, field


@dataclass(slots=True)
class GAConfig:
    """
    Configuration for the AETHER Evolutionary Swarm.
    Tuned for high diversity and robust global maximum discovery.
    """
    # --- System & Evolution Limits ---
    population_size: int = 25
    generations: int = 15         # FIX: bumped from 15 to 30 so swarm runs full lifecycle
    patience: int = 4               # Early stopping: halt if no improvement for 8 generations

    # --- Genetic Algorithm (GA) Operators ---
    mutation_rate: float = 0.8    # 40% chance to force a new mutation at a causal position
    crossover_rate: float = 0.7    # 70% chance parents swap traits

    # --- Particle Swarm (PSO) Dynamics ---
    inertia_weight: float = 0.4    # w: Momentum (lowered to prevent over-shooting)
    cognitive_coeff: float = 1.8   # c1: Favour personal exploration
    social_coeff: float = 0.8      # c2: Reduced to prevent early collapse

    # --- Dynamic Attributes ---
    swarm_size: int = field(init=False)

    # FIX: stagnation_limit is an alias for patience.
    # pso_optimizer.py references self.config.stagnation_limit.
    # Rather than rename patience everywhere, we expose both names
    # pointing to the same value via __post_init__.
    stagnation_limit: int = field(init=False)

    def __post_init__(self):
        """
        Automatically runs right after the class is initialized.
        Validates parameters and syncs dependent variables.
        """
        # Sync swarm size to match GA population
        self.swarm_size = self.population_size

        # Sync stagnation_limit to patience so both names work
        self.stagnation_limit = self.patience

        # Safety validation
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("Crossover rate must be between 0.0 and 1.0")