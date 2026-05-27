import torch

MODEL_NAME = "readerbench/RoGPT2-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"