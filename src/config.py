import torch

# Available models for analysis
AVAILABLE_MODELS = {
    "RoGPT2-base": {
        "name": "readerbench/RoGPT2-base",
        "language": "Romanian",
        "description": "RoGPT2 Base - Trained on Romanian text"
    },
    "RoGPT2-medium": {
        "name": "readerbench/RoGPT2-medium",
        "language": "Romanian",
        "description": "RoGPT2 Medium - Larger Romanian model"
    },
    "GPT2": {
        "name": "gpt2",
        "language": "English",
        "description": "Standard GPT2 - For comparison (English)"
    }
}

DEFAULT_MODEL = "RoGPT2-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"