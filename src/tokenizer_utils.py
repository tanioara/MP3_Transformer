from model_loader import tokenizer

def tokenize(text):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return tokens, inputs


def format_tokens_for_display(tokens):
    """
    Formatează tokenii pentru afișare ușoară, substituind 
    caracterele speciale din BPE cu reprezentări mai clare.
    """
    formatted = []
    for token in tokens:
        # Înlocuiți Ġ (spațiu BPE) cu [SPACE] pentru claritate
        if token.startswith("Ġ"):
            formatted.append(f"[SPACE]{token[1:]}")
        # Tokenii speciali rămân cum sunt (de ex: [CLS], [SEP])
        elif token.startswith("[") and token.endswith("]"):
            formatted.append(token)
        # Tokenii obișnuiți
        else:
            formatted.append(token)
    return formatted