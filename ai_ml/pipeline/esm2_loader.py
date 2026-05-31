import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_esm2_model(model_name="facebook/esm2_t30_150M_UR50D", quantize=True):
    """
    Loads the ESM-2 150M model and tokenizer from Hugging Face.
    Applies INT8 dynamic quantization for optimized CPU inference.
    """
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading raw model {model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    if quantize:
        print("Applying INT8 dynamic quantization (CPU optimization)...")
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.eval() 
    print("ESM-2 model successfully loaded and quantized.\n")
    
    return tokenizer, model
