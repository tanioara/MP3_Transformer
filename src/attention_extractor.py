from model_loader import tokenizer, model
from config import DEVICE

def extract_attentions(text):

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True
        )

    return outputs.attentions, tokens

import numpy as np

def top_attended_tokens(matrix, tokens, token_index, top_k=5):

    scores = matrix[token_index]

    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        (tokens[i], float(scores[i]))
        for i in top_idx
    ]