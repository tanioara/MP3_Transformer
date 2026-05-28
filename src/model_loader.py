from transformers import AutoTokenizer, AutoModelForCausalLM
from config import DEFAULT_MODEL, AVAILABLE_MODELS, DEVICE

# Global model and tokenizer (will be initialized with default)
tokenizer = None
model = None
current_model_name = None


def load_model(model_key):
    """
    Load a model and tokenizer by model key.
    
    Args:
        model_key: Key from AVAILABLE_MODELS dict (e.g., "RoGPT2-base")
    
    Returns:
        tuple: (tokenizer, model, model_name)
    """
    global tokenizer, model, current_model_name
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_name = AVAILABLE_MODELS[model_key]["name"]
    
    print(f"Loading model: {model_key} ({model_name})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        attn_implementation="eager"
    )
    
    model.to(DEVICE)
    model.eval()
    
    current_model_name = model_key
    
    print(f"Model loaded successfully!")
    
    return tokenizer, model, model_key


# Initialize with default model
load_model(DEFAULT_MODEL)