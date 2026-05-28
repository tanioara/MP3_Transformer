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


def generate_text_with_head_mask(prompt, layer_to_mask, head_to_mask, max_new_tokens=30):
    inputs = get_inputs(prompt)

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    head_mask = torch.ones(num_layers, num_heads).to(DEVICE)

    # dezactivăm head-ul ales
    head_mask[layer_to_mask, head_to_mask] = 0

    generated_ids = inputs["input_ids"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids,
                head_mask=head_mask,
                return_dict=True
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )