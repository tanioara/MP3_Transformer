import streamlit as st

from generation import generate_text
from attention_extractor import extract_attentions 



st.title("RoGPT2 Attention Analyzer")


prompt = st.text_area(
    "Enter Romanian text:",
    "Banca a aprobat creditul."
)


# RUN MODEL
if st.button("Generate + Analyze"):

    # 1. Generate text
    result = generate_text(prompt)

    st.subheader("Generated Text")
    st.write(result)

    # 2. Extract attention + tokens
    attentions, tokens = extract_attentions(prompt)

    st.subheader("Tokens")

    st.write(tokens)

    # LAYER + HEAD SELECTION
    layer = st.slider(
        "Layer",
        0,
        len(attentions) - 1,
        0
    )

    head = st.slider(
        "Head",
        0,
        attentions[0].shape[1] - 1,
        0
    )

    # TOKEN SELECTION
    token_index = st.selectbox(
        "Select token",
        list(range(len(tokens))),
        format_func=lambda i: f"{i}: {tokens[i]}"
    )


    # ATTENTION MATRIX
    matrix = attentions[layer][0, head].detach().cpu().numpy()


    # TOP TOKENS
    import numpy as np

    scores = matrix[token_index]
    top_idx = np.argsort(scores)[::-1][:5]

    st.subheader("Top attended tokens")

    for i in top_idx:
        st.write(tokens[i], float(scores[i]))