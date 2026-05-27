import torch

from model_loader import tokenizer, model
from config import DEVICE
from model_loader import tokenizer, model
from config import DEVICE

def get_inputs(prompt):
    return tokenizer(prompt, return_tensors="pt").to(DEVICE)

def generate_text(prompt, max_new_tokens=30):

    inputs = get_inputs(prompt)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )