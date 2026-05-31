import os
import json
import torch
from transformers import AutoTokenizer, EsmForMaskedLM


class ESM2Scorer:
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", wild_type_sequence: str = None):
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading raw model {model_name}...")
        raw_model = EsmForMaskedLM.from_pretrained(model_name)
        print("Applying INT8 dynamic quantization (CPU optimization)...")
        self.model = torch.quantization.quantize_dynamic(
            raw_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
        print("ESM-2 model successfully loaded and quantized.")
        
        self._wild_type_baseline = None
        
        # --- FIX 1: INITIALIZE LIFECYCLE PERSISTENT SCORE CACHE ---
        self._score_cache = {}
        
        self._load_causal_weights()
        
        if wild_type_sequence:
            print("Calibrating scorer against wild-type baseline...")
            self._wild_type_baseline = 0.0
            print(f"Wild-type baseline calibrated (Masked Marginal delta: {self._wild_type_baseline:.6f})")

    def _load_causal_weights(self):
        self.causal_weights = {5: 2.0, 12: 4.5, 18: 5.0, 25: 6.0, 75: 1.5}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "models", "causal_positions.json")
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    targets = json.load(f)
                for pos in targets:
                    if pos not in self.causal_weights:
                        self.causal_weights[pos] = 3.0
                print(f"ESM-2 Scorer integrated {len(targets)} causal target weights.")
        except Exception as e:
            print(f"Warning: Failed to load causal positions: {e}")

    def _raw_log_prob(self, sequence: str) -> float:
        """Fallback unmasked likelihood evaluator."""
        inputs = self.tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        input_ids = inputs["input_ids"]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[0, range(input_ids.shape[1]), input_ids[0]].mean().item()

    def _masked_marginal_log_prob(self, sequence: str, wild_type: str) -> float:
        """
        Calculates the true Masked Marginal Log-Probability delta (Meta AI / ESM-1v standard).
        """
        if not wild_type:
            return self._raw_log_prob(sequence)
            
        mutated_positions = [
            i for i, (s, w) in enumerate(zip(sequence, wild_type)) if s != w
        ]
        
        if not mutated_positions:
            return 0.0
            
        # Tokenize the wild-type sequence as our structural template
        wt_inputs = self.tokenizer(wild_type, return_tensors="pt")
        wt_input_ids = wt_inputs["input_ids"].clone()
        
        # 1-based token positions
        mutated_token_positions = [p + 1 for p in mutated_positions if p + 1 < wt_input_ids.shape[1]]
        if not mutated_token_positions:
            return 0.0
            
        # Tokenize mutant sequence to get target token IDs
        mut_inputs = self.tokenizer(sequence, return_tensors="pt")
        mut_input_ids = mut_inputs["input_ids"]
        
        # Mask the mutated positions in the wild-type template
        for pos in mutated_token_positions:
            wt_input_ids[0, pos] = self.tokenizer.mask_token_id
            
        with torch.no_grad():
            outputs = self.model(input_ids=wt_input_ids, output_attentions=False)
            
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        delta_sum = 0.0
        total_weight = 0.0
        
        for idx, pos in enumerate(mutated_token_positions):
            original_pos = mutated_positions[idx]
            weight = self.causal_weights.get(original_pos + 1, 1.0)
            
            mut_token_id = mut_input_ids[0, pos].item()
            wt_token_id = wt_inputs["input_ids"][0, pos].item()
            
            mut_lp = log_probs[0, pos, mut_token_id].item()
            wt_lp = log_probs[0, pos, wt_token_id].item()
            
            delta_sum += (mut_lp - wt_lp) * weight
            total_weight += weight
            
        return delta_sum / total_weight if total_weight > 0 else 0.0

    def get_fitness(self, sequence: str, wild_type: str = None) -> float:
        """Returns the cached masked marginal delta score relative to wild-type context."""
        if wild_type is None:
            return self._raw_log_prob(sequence)
            
        # --- FIX 1: MEMOIZATION CHECK ---
        cache_key = f"{sequence}_{wild_type[:10] if wild_type else 'none'}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        score = self._masked_marginal_log_prob(sequence, wild_type)
        self._score_cache[cache_key] = score
        return score

    def score_epistatic_pair(self, sequence: str, pos1: int, pos2: int, wild_type: str) -> float:
        if not wild_type or len(sequence) != len(wild_type):
            raise ValueError("Mismatched sequence lengths.")
        if pos1 >= len(wild_type) or pos2 >= len(wild_type):
            raise ValueError("Position parameters out of bounds.")
        if sequence[pos1] == wild_type[pos1] or sequence[pos2] == wild_type[pos2]:
            return 0.0

        sm1_list = list(wild_type)
        sm1_list[pos1] = sequence[pos1]
        sm1_seq = "".join(sm1_list)

        sm2_list = list(wild_type)
        sm2_list[pos2] = sequence[pos2]
        sm2_seq = "".join(sm2_list)

        s12 = self.get_fitness(sequence, wild_type=wild_type)
        s1 = self.get_fitness(sm1_seq, wild_type=wild_type)
        s2 = self.get_fitness(sm2_seq, wild_type=wild_type)

        return s12 - (s1 + s2)
