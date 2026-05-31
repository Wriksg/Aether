import os
import sys
import json
import numpy as np

# --- DYNAMIC PATH RESOLUTION BOILERPLATE ---
# Since this file is in ai_ml/tests/, we traverse 3 levels up to find the project root (AETHER).
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_ml.brain.inference_engine import InferenceEngine
from ai_ml.brain.ga_config import GAConfig


def validate_score_distribution():
    print("==================================================")
    print("      AETHER SWARM SCORE SPREAD VALIDATOR         ")
    print("==================================================")

    # 1. Instantiate engine with a lightweight active site search space
    engine = InferenceEngine()
    demo_config = GAConfig(
        population_size=15,
        generations=5,
        patience=2
    )
    engine.config = demo_config

    # Active site test fragment
    demo_fragment = "MQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSR"

    print("Executing active swarm search to generate population distribution...")
    json_payload = engine.run(demo_fragment)
    data = json.loads(json_payload)

    candidates = data.get("candidates", [])

    # 2. Extract sequences and scores
    scores = [c["score"] for c in candidates]
    sequences = [c["sequence"] for c in candidates]

    if len(scores) < 10:
        print(f"CRITICAL ERROR: Swarm only returned {len(scores)} candidates (expected 10).")
        sys.exit(1)

    # 3. Compute Statistical metrics
    unique_seqs = set(sequences)
    unique_count = len(unique_seqs)

    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    score_range = max_score - min_score

    # Count variants that successfully beat the baseline (score > 0.0)
    positive_mutants = sum(1 for s in scores if s > 0.0)

    # 4. Enforce Strict Statistical Validation Thresholds
    thresholds_passed = True
    errors = []

    # Check 1: Population diversity (Standard Deviation threshold)
    if std_dev <= 0.01:
        thresholds_passed = False
        errors.append(f"Premature Convergence: Standard deviation is too low ({std_dev:.6f} <= 0.010).")

    # Check 2: Absolute Sequence Uniqueness
    if unique_count != len(candidates):
        thresholds_passed = False
        errors.append(
            f"Duplicate Sequences: Found non-unique genomes ({unique_count} unique out of {len(candidates)}).")

    # Check 3: Search Capability (At least one variant must beat baseline)
    if max_score <= 0.0:
        thresholds_passed = False
        errors.append(
            f"No Improvement: Max candidate score ({max_score:.6f}) failed to exceed wild-type baseline (0.00).")

    # 5. Output Report
    print("\n--------------------------------------------------")
    print("            MATHEMATICAL EVALUATION REPORT        ")
    print("--------------------------------------------------")
    print(f"Candidates Tested:  {len(scores)}")
    print(f"Unique Genomes:     {unique_count}/10")
    print(f"Maximum Score:      {max_score:+.6f} delta log-prob")
    print(f"Minimum Score:      {min_score:+.6f} delta log-prob")
    print(f"Score Range:        {score_range:.6f}")
    print(f"Population Mean:    {mean_score:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Positive Variants:  {positive_mutants} out of {len(scores)}")
    print("--------------------------------------------------")

    if thresholds_passed:
        print("RESULT: SUCCESS - All score spread validation thresholds passed.")
        print("==================================================")
        sys.exit(0)
    else:
        print("RESULT: FAILED - Scientific thresholds violated:")
        for err in errors:
            print(f"  * {err}")
        print("==================================================")
        sys.exit(1)


if __name__ == "__main__":
    validate_score_distribution()