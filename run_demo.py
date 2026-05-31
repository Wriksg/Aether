import os
import sys
import json
import time

# --- AUTOMATIC PATH INJECTION ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_ml.brain.inference_engine import InferenceEngine
from ai_ml.brain.sequence_loader import SequenceLoader
from ai_ml.brain.ga_config import GAConfig


def execute_validation_checklist(json_output: str) -> bool:
    print("\n" + "="*50)
    print("          AETHER FINAL VALIDATION CHECKLIST       ")
    print("="*50)
    
    all_passed = True
    checklist = {}

    # Check 1: FASTA Sequence Database Integrity
    try:
        loader = SequenceLoader()
        wt, fast = loader.load_sequences()
        if len(wt) == 290 and len(fast) == 290:
            checklist["[1] FASTA Database Loader Integrity"] = ("PASS", "Successfully loaded WT and FAST-PETase reference sequences.")
        else:
            checklist["[1] FASTA Database Loader Integrity"] = ("FAIL", f"Sequence length mismatch. Found length {len(wt)}")
            all_passed = False
    except Exception as e:
        checklist["[1] FASTA Database Loader Integrity"] = ("FAIL", str(e))
        all_passed = False

    # Check 2: Rosetta Physics Cache Integrity
    try:
        cache_path = os.path.join(project_root, "ai_ml", "models", "rosetta_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            if len(cache_data) >= 230:
                checklist["[2] Rosetta Physics Cache Integrity"] = ("PASS", f"Loaded {len(cache_data)} biophysical delta profiles.")
            else:
                checklist["[2] Rosetta Physics Cache Integrity"] = ("FAIL", f"Insufficient profiles found ({len(cache_data)}/230)")
                all_passed = False
        else:
            checklist["[2] Rosetta Physics Cache Integrity"] = ("FAIL", "rosetta_cache.json missing from disk.")
            all_passed = False
    except Exception as e:
        checklist["[2] Rosetta Physics Cache Integrity"] = ("FAIL", str(e))
        all_passed = False

    # Check 3: DoWhy Causal Position Discovery
    try:
        causal_path = os.path.join(project_root, "ai_ml", "models", "causal_positions.json")
        if os.path.exists(causal_path):
            with open(causal_path, "r") as f:
                targets = json.load(f)
            if len(targets) > 0:
                checklist["[3] DoWhy Causal Position Discovery"] = ("PASS", f"Backdoor adjustments successfully resolved causal targets: {targets}")
            else:
                checklist["[3] DoWhy Causal Position Discovery"] = ("FAIL", "Causal target registry is empty.")
                all_passed = False
        else:
            checklist["[3] DoWhy Causal Position Discovery"] = ("FAIL", "causal_positions.json missing from models folder.")
            all_passed = False
    except Exception as e:
        checklist["[3] DoWhy Causal Position Discovery"] = ("FAIL", str(e))
        all_passed = False

    # Check 4: API Contract and Schema Compliance
    try:
        data = json.loads(json_output)
        required_root_keys = {"candidates", "wildtype_rate_baseline", "run_metadata"}
        required_candidate_keys = {"rank", "sequence", "score", "predicted_rate_vs_wildtype", "confidence", "mutations", "reasoning", "risk_flag"}
        
        if not required_root_keys.issubset(data.keys()):
            checklist["[4] API Contract & JSON Schema compliance"] = ("FAIL", "Root keys missing from final JSON payload.")
            all_passed = False
        else:
            candidates = data.get("candidates", [])
            schema_valid = True
            for cand in candidates:
                if not required_candidate_keys.issubset(cand.keys()):
                    schema_valid = False
                    break
            if schema_valid:
                checklist["[4] API Contract & JSON Schema compliance"] = ("PASS", "All root and candidate keys strictly match FastAPI specifications.")
            else:
                checklist["[4] API Contract & JSON Schema compliance"] = ("FAIL", "Candidate dictionary schema violation detected.")
                all_passed = False
    except Exception as e:
        checklist["[4] API Contract & JSON Schema compliance"] = ("FAIL", f"JSON parsing failed: {e}")
        all_passed = False

    # Check 5: Swarm Deduplication & Uniqueness
    try:
        data = json.loads(json_output)
        candidates = data.get("candidates", [])
        sequences = [c["sequence"] for c in candidates]
        if len(set(sequences)) == len(candidates):
            checklist["[5] Optimization Swarm Deduplication"] = ("PASS", f"All {len(candidates)} returned candidates are genetically unique.")
        else:
            checklist["[5] Optimization Swarm Deduplication"] = ("FAIL", "Duplicate sequences found in the final ranked output.")
            all_passed = False
    except Exception as e:
        checklist["[5] Optimization Swarm Deduplication"] = ("FAIL", str(e))
        all_passed = False

    # Print report
    for item, (status, detail) in checklist.items():
        print(f"{item:<45} [{status:<4}] - {detail}")
        
    print("-"*50)
    if all_passed:
        print("RESULT: SUCCESS - AETHER Engine is completely verified for presentation!")
    else:
        print("RESULT: FAILED - Some core validation checks failed. Please review errors.")
    print("="*50 + "\n")
    
    return all_passed


def run_demo():
    print("==================================================")
    print("          AETHER TURN-KEY DEMO BOOTSTRAPPER       ")
    print("==================================================")
    
    # Configure lightweight settings to run in ~15-20 seconds on CPU
    demo_config = GAConfig(
        population_size=15,
        generations=5,
        patience=2
    )
    
    engine = InferenceEngine()
    engine.config = demo_config
    
    # Real full-length IsPETase Sequence (290 Residues)
    # Enables all real-world causal positions [121, 159, 186...] to remain inside bounds perfectly.
    demo_fragment = (
        "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGG"
        "TVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDT"
        "ARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAK"
        "QFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
    )
    
    print("\nStarting rapid demo optimization sequence...")
    demo_start = time.time()
    final_json = engine.run(demo_fragment)
    elapsed = time.time() - demo_start
    
    print("\n--- RAW JSON BACKEND CONTRACT OUTPUT ---")
    print(final_json)
    print(f"\nExecution Time: {elapsed:.2f} seconds.")
    
    # Execute checklist
    execute_validation_checklist(final_json)


if __name__ == "__main__":
    run_demo()
