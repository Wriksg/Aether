import os
import sys
import json
import pytest

# --- DYNAMIC PATH RESOLUTION BOILERPLATE ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_ml.brain.inference_engine import InferenceEngine
from ai_ml.brain.ga_config import GAConfig


def test_inference_engine_json_contract():
    """
    Verifies that the InferenceEngine output payload complies exactly
    with the expected FastAPI backend schema contract.
    """
    engine = InferenceEngine()

    # Configure a rapid execution profile (2 generations, 10 particles) to run in under 3 seconds
    demo_config = GAConfig(
        population_size=10,
        generations=2,
        patience=1
    )
    engine.config = demo_config

    # 50-AA fragment for fast CPU execution
    demo_fragment = "MQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSR"

    json_output = engine.run(demo_fragment)
    data = json.loads(json_output)

    # 1. Validate top-level keys
    assert "candidates" in data
    assert "wildtype_rate_baseline" in data
    assert "run_metadata" in data

    # 2. Validate metadata values
    metadata = data["run_metadata"]
    assert "generations_run" in metadata
    assert "swarm_size" in metadata
    assert "causal_positions_used" in metadata
    assert "execution_time_seconds" in metadata
    assert metadata["swarm_size"] == 10

    # 3. Validate candidates structure and bounds
    candidates = data["candidates"]
    assert len(candidates) > 0

    for cand in candidates:
        assert "rank" in cand
        assert "sequence" in cand
        assert "score" in cand
        assert "predicted_rate_vs_wildtype" in cand
        assert "confidence" in cand
        assert "mutations" in cand
        assert "reasoning" in cand
        assert "risk_flag" in cand

        # Sequence length must be perfectly preserved
        assert len(cand["sequence"]) == len(demo_fragment)