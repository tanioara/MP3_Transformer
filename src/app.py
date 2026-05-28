import streamlit as st
import numpy as np

from generation import generate_text
from attention_extractor import extract_attentions


st.title("RoGPT2 Attention Analyzer")

prompt = st.text_area(
    "Enter Romanian text:",
    "Banca a aprobat creditul."
)

if st.button("Generate + Analyze"):
    result = generate_text(prompt)
    tokens, attentions = extract_attentions(prompt)

    st.session_state["result"] = result
    st.session_state["tokens"] = tokens
    st.session_state["attentions"] = attentions
    st.session_state["prompt"] = prompt


if "attentions" in st.session_state:

    result = st.session_state["result"]
    tokens = st.session_state["tokens"]
    attentions = st.session_state["attentions"]

    st.subheader("Generated Text")
    st.write(result)

    st.subheader("Tokens")
    st.write(list(enumerate(tokens)))

    st.subheader("Debug attention shape")
    st.write("Number of layers:", len(attentions))
    st.write("Shape of first attention tensor:", attentions[0].shape)

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    layer = st.slider("Layer", 0, num_layers - 1, 0)
    head = st.slider("Head", 0, num_heads - 1, 0)

    token_index = st.selectbox(
        "Select token",
        list(range(len(tokens))),
        format_func=lambda i: f"{i}: {tokens[i]}"
    )

    matrix = attentions[layer][0, head].detach().cpu().numpy()

    scores = matrix[token_index]
    top_idx = np.argsort(scores)[::-1][:5]

    st.subheader("Top attended tokens")

    for i in top_idx:
        st.write(f"{tokens[i]}: {float(scores[i]):.4f}")

else:
    st.info("Enter a Romanian text and click Generate + Analyze.")