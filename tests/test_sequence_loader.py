import os
import sys

# --- DYNAMIC PATH RESOLUTION BOILERPLATE ---
# This ensures the project root directory is always in the Python path,
# preventing ModuleNotFoundError regardless of where or how you run this script.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import shutil
import unittest
from ai_ml.brain.sequence_loader import SequenceLoader, WILD_TYPE_PETASE, FAST_PETASE
from ai_ml.brain.inference_engine import InferenceEngine


class TestSequenceLoaderAndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We will use a temporary directory for sandboxing the file IO tests
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.temp_data_dir = os.path.join(base_dir, "data", "test_raw_sandbox")
        cls.loader = SequenceLoader(data_dir=cls.temp_data_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up sandbox environment after testing
        if os.path.exists(cls.temp_data_dir):
            shutil.rmtree(cls.temp_data_dir)

    def test_01_directory_and_fasta_auto_generation(self):
        """Verifies that SequenceLoader creates missing folders and FASTA defaults."""
        # Ensure sandbox directory doesn't exist yet
        if os.path.exists(self.temp_data_dir):
            shutil.rmtree(self.temp_data_dir)

        # Triggers ensure_fasta_exists internally
        wt_seq, fast_seq = self.loader.load_sequences()

        # Check file existence
        self.assertTrue(os.path.exists(self.loader.fasta_path))
        self.assertTrue(os.path.getsize(self.loader.fasta_path) > 0)

        # Validate sequence content lengths
        self.assertEqual(len(wt_seq), 290)
        self.assertEqual(len(fast_seq), 290)

    def test_02_biochemical_mutation_positions(self):
        """Verifies that the FAST-PETase mutations are correct and present at exact indices."""
        wt_seq, fast_seq = self.loader.load_sequences()

        # Define 1-based mutation positions and expected residues
        expected_mutations = {
            121: ("S", "E"),  # S121E
            186: ("D", "H"),  # D186H
            224: ("R", "Q"),  # R224Q
            233: ("N", "K"),  # N233K
            280: ("R", "A"),  # R280A
        }

        for pos_1based, (wt_aa, mut_aa) in expected_mutations.items():
            idx = pos_1based - 1
            # Check original residues in loaded wild-type
            self.assertEqual(
                wt_seq[idx],
                wt_aa,
                f"WT mismatch at index {idx} (1-based position {pos_1based}): Expected {wt_aa}, found {wt_seq[idx]}"
            )
            # Check mutated residues in loaded FAST-PETase
            self.assertEqual(
                fast_seq[idx],
                mut_aa,
                f"Mutation mismatch at index {idx} (1-based position {pos_1based}): Expected {mut_aa}, found {fast_seq[idx]}"
            )

    def test_03_inference_engine_integration_dry(self):
        """Dry-runs the InferenceEngine initialization to ensure SequenceLoader resolves inside it."""
        try:
            # Instantiate engine to verify it compiles and loads without path failures
            engine = InferenceEngine()
            self.assertIsNotNone(engine.config)
            self.assertIsNotNone(engine.physics_filter)
        except Exception as e:
            self.fail(f"InferenceEngine compilation/loading failed with error: {e}")


if __name__ == "__main__":
    unittest.main()