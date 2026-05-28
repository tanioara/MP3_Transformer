import torch
import torch.nn.functional as F

import model_loader
from config import DEVICE

def get_inputs(prompt):
    return model_loader.tokenizer(prompt, return_tensors="pt").to(DEVICE)

def generate_text(prompt, max_new_tokens=30, temperature=1.0, top_p=0.9):
    """
    Generate text with optional sampling for more variation.
    
    Args:
        prompt: Input text
        max_new_tokens: Number of tokens to generate
        temperature: Temperature for sampling (1.0 = normal, >1.0 = more random, <1.0 = more deterministic)
        top_p: Nucleus sampling parameter (0-1)
    """
    inputs = get_inputs(prompt)

    with torch.no_grad():
        output = model_loader.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )

    return model_loader.tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


def _sample_token(logits, temperature=1.0, top_p=0.9):
    """Sample a token from logits with temperature and top-p."""
    logits = logits.squeeze(0) if logits.dim() > 1 else logits
    
    if temperature == 0:
        # Greedy
        return torch.argmax(logits, dim=-1).unsqueeze(0)
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p (nucleus sampling)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumsum_probs > top_p
    sorted_indices_to_remove[0] = False  # Always keep top token
    
    sorted_logits[sorted_indices_to_remove] = -float('inf')
    
    # Sample
    probs = F.softmax(sorted_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1).item()
    
    return sorted_indices[next_token_idx].unsqueeze(0)


def generate_text_with_head_mask(prompt, layer_to_mask, head_to_mask, max_new_tokens=30, temperature=1.0, top_p=0.9):
    """
    Generate text with a specific head masked/ablated using manual token generation.
    This allows head_mask to work properly.
    """
    inputs = get_inputs(prompt)
    generated_ids = inputs["input_ids"]

    num_layers = model_loader.model.config.n_layer
    num_heads = model_loader.model.config.n_head

    head_mask = torch.ones(num_layers, num_heads).to(DEVICE)
    head_mask[layer_to_mask, head_to_mask] = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model_loader.model(
                input_ids=generated_ids,
                head_mask=head_mask,
                return_dict=True
            )

            logits = outputs.logits[:, -1, :]
            next_token_id = _sample_token(logits, temperature=temperature, top_p=top_p)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

    return model_loader.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )


def generate_text_with_multiple_heads_masked(prompt, heads_to_mask, max_new_tokens=30, temperature=1.0, top_p=0.9):
    """
    Generate text with multiple heads masked at once using manual token generation.
    
    Args:
        prompt: Input text
        heads_to_mask: List of tuples [(layer, head), (layer, head), ...]
        max_new_tokens: Number of tokens to generate
        temperature: Temperature for sampling
        top_p: Nucleus sampling parameter
    """
    inputs = get_inputs(prompt)
    generated_ids = inputs["input_ids"]

    num_layers = model_loader.model.config.n_layer
    num_heads = model_loader.model.config.n_head

    head_mask = torch.ones(num_layers, num_heads).to(DEVICE)

    # Dezactivează toate head-urile din lista
    for layer, head in heads_to_mask:
        head_mask[layer, head] = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model_loader.model(
                input_ids=generated_ids,
                head_mask=head_mask,
                return_dict=True
            )

            logits = outputs.logits[:, -1, :]
            next_token_id = _sample_token(logits, temperature=temperature, top_p=top_p)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

    return model_loader.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )


def get_head_importance_scores(prompt, max_new_tokens=30):
    """
    Calculate importance score for each head by ablating it individually.
    Returns a tensor of shape (num_layers, num_heads) with importance scores.
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    
    baseline_output = generate_text(prompt, max_new_tokens)
    baseline_tokens = tokenizer(baseline_output, return_tensors="pt")["input_ids"][0]
    
    importance_scores = torch.zeros(num_layers, num_heads)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            ablated_output = generate_text_with_head_mask(prompt, layer, head, max_new_tokens)
            ablated_tokens = tokenizer(ablated_output, return_tensors="pt")["input_ids"][0]
            
            # Calculate token-level difference (how many tokens changed)
            diff = torch.abs(baseline_tokens[:min(len(baseline_tokens), len(ablated_tokens))] - 
                           ablated_tokens[:min(len(baseline_tokens), len(ablated_tokens))]).float().mean()
            
            # Calculate text-level difference (string distance)
            str_diff = abs(len(baseline_output) - len(ablated_output)) / max(len(baseline_output), 1)
            
            importance_scores[layer, head] = diff.item() + str_diff
    
    return importance_scores


def get_head_entropy_scores(prompt):
    """
    Calculate entropy-based importance: how much does each head contribute to attention?
    Higher entropy = more important for attention distribution.
    """
    from attention_extractor import extract_attentions
    import torch.nn.functional as F
    
    tokens, attentions = extract_attentions(prompt)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    entropy_scores = torch.zeros(num_layers, num_heads)
    
    for layer in range(num_layers):
        attn_matrix = attentions[layer][0]  # (num_heads, seq_len, seq_len)
        
        for head in range(num_heads):
            head_attn = attn_matrix[head]  # (seq_len, seq_len)
            
            # Calculate entropy for each token's attention distribution
            eps = 1e-10
            entropy = -torch.sum(head_attn * torch.log(head_attn + eps), dim=-1).mean()
            
            entropy_scores[layer, head] = entropy.item()
    
    return entropy_scores