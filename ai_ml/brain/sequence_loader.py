import os
from typing import Tuple

# Standard high-fidelity 290-residue wild-type IsPETase precursor sequence (UniProt A0A0K8P6T7)
WILD_TYPE_PETASE = (
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYY"
    "PTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGT"
    "SSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAP"
    "VNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTR"
    "VSDFRTANCS"
)

# Apply the five mutations defining the FAST-PETase scaffold (S121E, D186H, R224Q, N233K, R280A)
# Map 1-based precursor residue numbers to 0-based indices:
# S121E -> Index 120
# D186H -> Index 185
# R224Q -> Index 223
# N233K -> Index 232
# R280A -> Index 279
_fast_petase_list = list(WILD_TYPE_PETASE)
_fast_petase_list[121 - 1] = 'E'
_fast_petase_list[186 - 1] = 'H'
_fast_petase_list[224 - 1] = 'Q'
_fast_petase_list[233 - 1] = 'K'
_fast_petase_list[280 - 1] = 'A'
FAST_PETASE = "".join(_fast_petase_list)


class SequenceLoader:
    def __init__(self, data_dir: str = None):
        """
        Initializes the sequence loader. Resolves paths relative to the project
        root structure if no data directory is explicitly provided.
        """
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.raw_dir = os.path.join(base_dir, "data", "raw")
        else:
            self.raw_dir = data_dir

        self.fasta_path = os.path.join(self.raw_dir, "uniprot_sequences.fasta")

    def ensure_fasta_exists(self):
        """
        Guarantees directory structure and writes high-fidelity FASTA default
        entries if they are missing, preventing cold-start IO failures.
        """
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)

        if not os.path.exists(self.fasta_path):
            print(f"FASTA file not found. Generating default high-fidelity sequences at: {self.fasta_path}")
            with open(self.fasta_path, "w") as f:
                f.write(
                    ">sp|A0A0K8P6T7|PETH_IDESA Poly(ethylene terephthalate) hydrolase OS=Ideonella sakaiensis OX=1547922 GN=ISF6_4831 PE=1 SV=1\n")
                f.write(f"{WILD_TYPE_PETASE}\n")
                f.write(
                    ">engineered|FAST-PETase ML-optimized PET hydrolase variant (S121E, D186H, R224Q, N233K, R280A)\n")
                f.write(f"{FAST_PETASE}\n")

    def load_sequences(self) -> Tuple[str, str]:
        """
        Loads the wildtype IsPETase and the FAST-PETase sequences from the FASTA file.
        Falls back securely to high-fidelity biological constants if parsing fails.

        Returns:
            Tuple[str, str]: (wild_type_sequence, fast_petase_sequence)
        """
        self.ensure_fasta_exists()

        wt_seq = None
        fast_seq = None

        try:
            with open(self.fasta_path, "r") as f:
                current_header = ""
                current_seq = []

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if current_header:
                            seq_str = "".join(current_seq)
                            if "A0A0K8P6T7" in current_header or "PETH_IDESA" in current_header:
                                wt_seq = seq_str
                            elif "FAST-PETase" in current_header:
                                fast_seq = seq_str
                        current_header = line
                        current_seq = []
                    else:
                        current_seq.append(line)

                # Process the final record in file
                if current_header and current_seq:
                    seq_str = "".join(current_seq)
                    if "A0A0K8P6T7" in current_header or "PETH_IDESA" in current_header:
                        wt_seq = seq_str
                    elif "FAST-PETase" in current_header:
                        fast_seq = seq_str

        except Exception as e:
            print(f"Warning: Failed to parse sequence file: {e}. Utilizing embedded safety constants.")

        # Safe fallback checks
        if not wt_seq:
            wt_seq = WILD_TYPE_PETASE
        if not fast_seq:
            fast_seq = FAST_PETASE

        return wt_seq, fast_seq