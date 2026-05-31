import os
import sys
import pytest

# --- DYNAMIC PATH RESOLUTION BOILERPLATE ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_ml.brain.physics_filter import PhysicsFilter


def test_physics_filter_cache_loading():
    """
    Verifies that PhysicsFilter loads and correctly subtracts cached
    Rosetta pocket energy penalties from baseline scorer inputs.
    """
    pf = PhysicsFilter()

    # Check if the cache contains cached entries (built during Task 2)
    if pf.delta_cache:
        assert len(pf.delta_cache) > 0

        # Test subtraction math using a real cached key
        sample_seq = list(pf.delta_cache.keys())[0]
        penalty = pf.delta_cache[sample_seq]

        base_score = 1.50
        expected_score = base_score - penalty

        assert pf.apply_penalty(sample_seq, base_score) == pytest.approx(expected_score)
    else:
        # Secure fallback behavior check if cache is empty
        assert pf.apply_penalty("MQAAV", 1.50) == 1.50


def test_physics_filter_unknown_sequence():
    """
    Verifies that unknown sequences receive 0.0 penalty (passthrough mode).
    """
    pf = PhysicsFilter()
    base_score = 2.0

    # Generate an arbitrary sequence string that is not in the cache
    unknown_seq = "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"

    # The score should remain unpenalized (subtraction of 0.0)
    assert pf.apply_penalty(unknown_seq, base_score) == base_score