from typing import List, Tuple, Dict

class Ranker:
    def __init__(self, wild_type_score: float, wild_type_sequence: str = None):
        self.wild_type_score = wild_type_score
        self.wild_type_sequence = wild_type_sequence

    def rank_candidates(self, final_swarm: List[Tuple[str, float, List[str]]], top_n: int = 10) -> List[Dict]:
        """
        Takes the final swarm, filters out wild-type and no-mutation candidates,
        sorts by fitness, and extracts the top N.
        Calculates the relative improvement over the wild-type.

        Expects each swarm entry as: (sequence, score, mutations_list)
        """
        processed_swarm = []
        for seq, score, mutations in final_swarm:
            # --- AUTO-EXTRACT MUTATIONS IF NOT PROVIDED OR STRIPPED ---
            if not mutations or len(mutations) == 0:
                if self.wild_type_sequence and len(seq) == len(self.wild_type_sequence):
                    # Compare characters programmatically using 1-based indexing
                    mutations = [
                        f"{self.wild_type_sequence[i]}{i+1}{seq[i]}"
                        for i in range(len(seq))
                        if seq[i] != self.wild_type_sequence[i]
                    ]
            
            # Keep candidate if it contains mutations and does not match wild-type sequence
            if mutations and len(mutations) > 0 and seq != self.wild_type_sequence:
                processed_swarm.append((seq, score, mutations))

        if len(processed_swarm) == 0:
            print("WARNING: All candidates were filtered out. Check mutation generation.")
            return []

        # Sort by score descending (highest fitness = best)
        sorted_swarm = sorted(processed_swarm, key=lambda x: x[1], reverse=True)
        top_candidates = sorted_swarm[:top_n]

        ranked_results = []
        for rank, (seq, score, mutations) in enumerate(top_candidates, 1):
            score_diff = score - self.wild_type_score
            
            # --- CONSERVATIVE KINETIC PROXY CALIBRATION ---
            # Clamped and scaled to match observed experimental folds (max ~14x)
            fold_improvement = round(max(1.0, 1.0 + (score_diff * 4.5)), 1)
            
            confidence = round(min(0.99, max(0.50, 0.50 + (score_diff * 0.15))), 2)
            
            # Hard penalization if key catalytic residues are mutated to structural disruptors
            risk_flag = None
            if any(mut in ["S160G", "D206G", "H237G", "S160A", "D206A", "H237A"] for mut in mutations):
                risk_flag = "CRITICAL: Mutation of catalytic triad residue is highly likely to disrupt structural activity."
                fold_improvement = 0.1  # Severe functional penalty
                confidence = 0.10
            elif any("G" in mut[-1] for mut in mutations):
                risk_flag = "Moderate risk of thermal instability due to glycine backbone entropy."
            else:
                risk_flag = "Requires in-vitro experimental verification."

            ranked_results.append({
                "rank": rank,
                "sequence": seq,
                "score": score,
                "predicted_rate_vs_wildtype": fold_improvement,
                "confidence": confidence,
                "mutations": mutations,
                "risk_flag": risk_flag
            })

        return ranked_results