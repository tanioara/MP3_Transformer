import pandas as pd

from generation import generate_text
from attention_extractor import extract_attentions


def find_token_index(tokens, search_text):
    """
    Caută primul token care conține textul dat.
    Exemplu: search_text='ea' poate găsi tokenul 'Ġea'.
    """
    search_text = search_text.lower()

    for i, token in enumerate(tokens):
        clean_token = token.replace("Ġ", "").lower()

        if search_text in clean_token:
            return i

    return None


def get_head_ranking(attentions, source_index, target_index, top_k=10):
    """
    Calculează top head-uri pentru relația:
    source_token -> target_token
    """
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    ranking = []

    for layer in range(num_layers):
        for head in range(num_heads):
            matrix = attentions[layer][0, head].detach().cpu().numpy()
            score = float(matrix[source_index, target_index])

            ranking.append({
                "Layer": layer,
                "Head": head,
                "Attention score": score
            })

    ranking_df = pd.DataFrame(ranking)
    ranking_df = ranking_df.sort_values(
        by="Attention score",
        ascending=False
    ).reset_index(drop=True)

    return ranking_df.head(top_k)


def analyze_language_example(text, source_word, target_word):
    """
    Analizează un text:
    - generează output
    - extrage tokenii
    - găsește tokenul sursă și tokenul țintă
    - calculează top head-uri pentru relația source -> target
    """
    generated_text = generate_text(text)
    tokens, attentions = extract_attentions(text)

    source_index = find_token_index(tokens, source_word)
    target_index = find_token_index(tokens, target_word)

    result = {
        "text": text,
        "generated_text": generated_text,
        "tokens": tokens,
        "num_tokens": len(tokens),
        "source_word": source_word,
        "target_word": target_word,
        "source_index": source_index,
        "target_index": target_index,
        "ranking": None
    }

    if source_index is not None and target_index is not None:
        result["ranking"] = get_head_ranking(
            attentions,
            source_index,
            target_index
        )

    return result


def compare_ro_en(
    ro_text,
    en_text,
    ro_source_word,
    ro_target_word,
    en_source_word,
    en_target_word
):
    """
    Compară română vs engleză pentru două propoziții echivalente.
    """
    ro_result = analyze_language_example(
        ro_text,
        ro_source_word,
        ro_target_word
    )

    en_result = analyze_language_example(
        en_text,
        en_source_word,
        en_target_word
    )

    summary = pd.DataFrame([
        {
            "Language": "Romanian",
            "Text": ro_text,
            "Number of tokens": ro_result["num_tokens"],
            "Source token index": ro_result["source_index"],
            "Target token index": ro_result["target_index"]
        },
        {
            "Language": "English",
            "Text": en_text,
            "Number of tokens": en_result["num_tokens"],
            "Source token index": en_result["source_index"],
            "Target token index": en_result["target_index"]
        }
    ])

    return ro_result, en_result, summary