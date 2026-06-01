import math
from typing import Dict, List


class ReasoningGenerator:
    def __init__(self, wild_type: str, causal_positions: List[int]):
        self.wild_type = wild_type
        self.causal_positions = causal_positions

        # Detailed amino acid profiles for biochemically realistic reasoning cards
        self.AMINO_ACID_PROFILES = {
            "A": {
                "mechanism": "places a small hydrophobic alanine",
                "effect": "minimizes steric hindrance while maintaining a hydrophobic pocket contact",
                "risk": None
            },
            "C": {
                "mechanism": "places a cysteine",
                "effect": "introduces potential for localized disulfide bond formation with nearby residues",
                "risk": "Risk of forming aberrant disulfide bonds under non-reducing conditions."
            },
            "D": {
                "mechanism": "introduces a negatively charged aspartate",
                "effect": "creates new electrostatic interactions that stabilize substrate orientation in the binding cleft",
                "risk": "May alter local pH sensitivity near the active site."
            },
            "E": {
                "mechanism": "introduces a negatively charged glutamate",
                "effect": "adds a flexible carboxylate chain that can participate in crucial hydrogen bonding or salt-bridge networks",
                "risk": None
            },
            "F": {
                "mechanism": "introduces a bulky aromatic phenylalanine",
                "effect": "occupies empty hydrophobic pockets, restricting local conformational fluctuations",
                "risk": "Slight risk of steric clashes if surrounding packing is tight."
            },
            "G": {
                "mechanism": "removes the side chain entirely via glycine substitution",
                "effect": "introduces backbone flexibility at this position, allowing the binding pocket to adopt a wider-open conformation for PET entry",
                "risk": "Moderate risk of thermal instability due to increased backbone entropy."
            },
            "H": {
                "mechanism": "places a histidine imidazole ring",
                "effect": "acts as a potential proton transfer helper or a metal-coordination site to stabilize nearby transition states",
                "risk": None
            },
            "I": {
                "mechanism": "introduces isoleucine at this position",
                "effect": "fills a hydrophobic cavity with a beta-branched side chain, increasing structural rigidity around the binding cleft",
                "risk": None
            },
            "K": {
                "mechanism": "introduces a flexible, positively charged lysine",
                "effect": "creates electrostatic surface contacts to enhance solubility and prevent aggregation",
                "risk": None
            },
            "L": {
                "mechanism": "substitutes leucine, a larger branched aliphatic",
                "effect": "strengthens hydrophobic core packing near the active site, improving thermal resilience",
                "risk": None
            },
            "M": {
                "mechanism": "places a flexible sulfur-containing methionine",
                "effect": "adds van der Waals contacts with the PET substrate while maintaining backbone flexibility",
                "risk": "Methionine is susceptible to oxidation under aerobic conditions."
            },
            "N": {
                "mechanism": "introduces a polar amide side chain",
                "effect": "adds a precise hydrogen bond donor-acceptor pair that coordinates with the water molecule network",
                "risk": None
            },
            "P": {
                "mechanism": "introduces a proline ring constraint",
                "effect": "rigidifies the local backbone into a fixed conformation that pre-organizes the active site for catalysis",
                "risk": "May shift optimal temperature window upward."
            },
            "Q": {
                "mechanism": "places a polar glutamine side chain",
                "effect": "extends a hydrogen-bonding network into the active site to stabilize catalytic intermediates",
                "risk": None
            },
            "R": {
                "mechanism": "places a positively charged arginine",
                "effect": "constructs a stable bidentate salt-bridge network with neighboring acidic residues",
                "risk": None
            },
            "S": {
                "mechanism": "places a small polar serine hydroxyl",
                "effect": "forms hydrogen bonds with the ester linkages of PET to pre-position the scissile bond",
                "risk": None
            },
            "T": {
                "mechanism": "places a threonine hydroxyl-methyl group",
                "effect": "creates a dual hydrogen bond donor-acceptor stabilizing the tetrahedral intermediate",
                "risk": "Requires in-vitro verification of hydrogen bond geometry."
            },
            "V": {
                "mechanism": "places a branched valine",
                "effect": "adds a compact hydrophobic wedge that tightens packing around the active site without blocking substrate access",
                "risk": None
            },
            "W": {
                "mechanism": "places a bulky aromatic tryptophan",
                "effect": "creates favorable pi-stacking interactions with the aromatic terephthalate rings of PET",
                "risk": "High risk of steric clashes if not accommodated by local structural relaxation."
            },
            "Y": {
                "mechanism": "places a tyrosine phenolic hydroxyl",
                "effect": "enables simultaneous pi-stacking and hydrogen-bonding to stabilize the binding of PET oligomers",
                "risk": "May shift pH optimum slightly."
            }
        }

        # Contextual mapping of physical positions to local functional regions
        self.POSITION_CONTEXT = {
            121: "near the oxyanion hole stabilization region",
            159: "in the substrate-binding pocket",
            186: "near the oxyanion hole stabilization region",
            206: "in the lid domain controlling substrate access",
            224: "in the lid domain controlling substrate access",
            233: "in the lid domain controlling substrate access",
            237: "in the lid domain controlling substrate access",
            280: "in the lid domain controlling substrate access"
        }

    def _build_reasoning(self, mutations: List[str], final_rate: float) -> tuple[str, str]:
        if not mutations:
            return "This is the wild-type baseline sequence. No mutations applied.", None

        sentences = []
        highest_risk = None

        for mut in mutations:
            wt_aa = mut[0]
            pos_str = "".join([char for char in mut if char.isdigit()])
            pos = int(pos_str)
            mut_aa = mut[-1]

            profile = self.AMINO_ACID_PROFILES.get(mut_aa, {
                "mechanism": f"substitutes {mut_aa}",
                "effect": f"alters local properties at position {pos}",
                "risk": None
            })
            context = self.POSITION_CONTEXT.get(pos, "near the active site cavity")

            sentence = f"{mut}: This substitution {profile['mechanism']} {context}. It {profile['effect']}."
            sentences.append(sentence)

            if profile.get("risk") and not highest_risk:
                highest_risk = profile["risk"]

        combined_reasoning = " | ".join(sentences)
        combined_reasoning += f" Combined, these changes are predicted to accelerate PET hydrolysis by {final_rate}x over wild-type."

        return combined_reasoning, highest_risk

    def generate_cards(self, top_candidates: List[Dict]) -> List[Dict]:
        """
        Analyzes the top candidates, extracts exact biological mutations,
        and formats them into the final JSON response.
        """
        output = []
        for i, cand in enumerate(top_candidates):
            seq = cand["sequence"]
            
            # Trust mutations already extracted by inference_engine converter
            mutations = cand.get("mutations", [])
            
            # Safety fallback: recompute from full sequence diff if empty
            if not mutations and self.wild_type:
                mutations = [
                    f"{self.wild_type[j]}{j+1}{seq[j]}"
                    for j in range(min(len(seq), len(self.wild_type)))
                    if seq[j] != self.wild_type[j]
                ]

            # Re-use the scientifically calibrated metrics from ranker
            rate_vs_wt = cand.get("predicted_rate_vs_wildtype", 1.0)
            confidence = cand.get("confidence", 0.5)
            delta_score = cand.get("score", 0.0)

            # Generate reasoning and risk flag dynamically
            reasoning, risk = self._build_reasoning(mutations, rate_vs_wt)
            
            # Merge risk flags: prefer critical triad flags from the ranker, fallback to residue risks
            final_risk = cand.get("risk_flag") if cand.get("risk_flag") else risk

            card = {
                "rank": i + 1,
                "sequence": seq,
                "score": delta_score,
                "predicted_rate_vs_wildtype": rate_vs_wt,
                "confidence": confidence,
                "mutations": mutations,
                "reasoning": reasoning,
                "risk_flag": final_risk
            }
            output.append(card)

        return output