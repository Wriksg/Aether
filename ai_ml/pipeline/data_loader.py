import os
import requests
import pandas as pd


class DataLoader:
    def __init__(self):
        # Ensure our raw data directory exists
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def fetch_uniprot_petase(self):
        """
        Surgically queries the UniProt REST API for PETase and MHETase sequences.
        """
        print("Connecting to UniProt REST API...")

        # Query specifically for plastic-degrading enzymes
        url = "https://rest.uniprot.org/uniprotkb/search?query=PETase OR MHETase&format=json&size=50"

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"CRITICAL ERROR: Failed to fetch data from UniProt: {e}")
            return None

        data = response.json()
        records = []

        print(f"Found {len(data.get('results', []))} candidate proteins. Extracting sequences...")

        for entry in data.get("results", []):
            # Safely extract deeply nested JSON data
            protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get(
                "value", "Unknown")
            organism = entry.get("organism", {}).get("scientificName", "Unknown")
            sequence = entry.get("sequence", {}).get("value", "")
            uniprot_id = entry.get("primaryAccession", "Unknown")

            # We only want entries that actually have a sequence attached
            if sequence:
                records.append({
                    "uniprot_id": uniprot_id,
                    "protein_name": protein_name,
                    "organism": organism,
                    "sequence": sequence,
                    "sequence_length": len(sequence)
                })

        # Save to CSV for the preprocessor to use later
        df = pd.DataFrame(records)
        output_path = os.path.join(self.raw_data_dir, "uniprot_petase_raw.csv")
        df.to_csv(output_path, index=False)

        print(f"\n✅ SUCCESS: Downloaded {len(df)} raw sequences.")
        print(f"📁 Saved to: {output_path}")

        return output_path


if __name__ == "__main__":
    print("--- Starting AETHER Data Loader ---")
    loader = DataLoader()
    loader.fetch_uniprot_petase()
