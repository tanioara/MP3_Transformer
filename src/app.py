import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from generation import generate_text, generate_text_with_head_mask
from attention_extractor import extract_attentions
from language_comparison import compare_ro_en


st.title("RoGPT2 Attention Analyzer")

st.write(
    "Aplicație pentru analiza attention heads într-un model RoGPT2. "
    "Poți introduce un text în română, vizualiza atenția dintre tokeni "
    "și testa ce se întâmplă când dezactivezi anumite head-uri."
)


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
    prompt = st.session_state["prompt"]

    st.subheader("Generated Text")
    st.write(result)

    st.subheader("Tokens")
    st.write(list(enumerate(tokens)))

    st.subheader("Debug attention shape")
    st.write("Number of layers:", len(attentions))
    st.write("Shape of first attention tensor:", attentions[0].shape)

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    # ---------------------------------------------------------
    # ATTENTION EXPLORER
    # ---------------------------------------------------------

    st.subheader("Attention Explorer")

    layer = st.slider("Layer", 0, num_layers - 1, 0)
    head = st.slider("Head", 0, num_heads - 1, 0)

    token_index = st.selectbox(
        "Select token",
        list(range(len(tokens))),
        format_func=lambda i: f"{i}: {tokens[i]}"
    )

    matrix = attentions[layer][0, head].detach().cpu().numpy()

    # ---------------------------------------------------------
    # ATTENTION HEATMAP
    # ---------------------------------------------------------

    st.subheader("Attention Heatmap")

    st.write(
        "Heatmap-ul arată câtă atenție acordă fiecare token celorlalți tokeni. "
        "Rândul reprezintă tokenul curent, iar coloana reprezintă tokenul la care se uită."
    )

    fig = px.imshow(
        matrix,
        x=tokens,
        y=tokens,
        labels=dict(
            x="Token attended to",
            y="Current token",
            color="Attention"
        ),
        title=f"Attention Heatmap - Layer {layer}, Head {head}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TOP ATTENDED TOKENS
    # ---------------------------------------------------------

    scores = matrix[token_index]
    top_idx = np.argsort(scores)[::-1][:5]

    st.subheader("Top attended tokens")

    st.write(
        "Lista arată tokenii la care tokenul selectat acordă cea mai multă atenție "
        "în layer-ul și head-ul ales."
    )

    selected_token = tokens[token_index]

    for i in top_idx:
        st.write(
            f"Tokenul **{selected_token}** se uită la **{tokens[i]}** "
            f"cu scorul `{float(scores[i]):.4f}`"
        )

    # ---------------------------------------------------------
    # HEAD RELEVANCE RANKING
    # ---------------------------------------------------------

    st.subheader("Head relevance ranking")

    st.write(
        "Alege două tokenuri pentru a vedea care layer/head acordă cea mai mare atenție "
        "relației dintre ele."
    )

    col1, col2 = st.columns(2)

    with col1:
        source_token = st.selectbox(
            "Token care acordă atenție",
            list(range(len(tokens))),
            format_func=lambda i: f"{i}: {tokens[i]}",
            key="source_token"
        )

    with col2:
        target_token = st.selectbox(
            "Token urmărit",
            list(range(len(tokens))),
            format_func=lambda i: f"{i}: {tokens[i]}",
            key="target_token"
        )

    ranking = []

    for l in range(num_layers):
        for h in range(num_heads):
            attn_matrix = attentions[l][0, h].detach().cpu().numpy()
            score = float(attn_matrix[source_token, target_token])

            ranking.append({
                "Layer": l,
                "Head": h,
                "Attention score": score
            })

    ranking_df = pd.DataFrame(ranking)
    ranking_df = ranking_df.sort_values(
        by="Attention score",
        ascending=False
    ).reset_index(drop=True)

    st.write(
        f"Relație analizată: **{tokens[source_token]} → {tokens[target_token]}**"
    )

    st.dataframe(ranking_df.head(10), use_container_width=True)

    # ---------------------------------------------------------
    # HEAD ABLATION EXPERIMENT
    # ---------------------------------------------------------

    st.subheader("Head Ablation Experiment")

    st.write(
        "În această secțiune dezactivăm un attention head și comparăm textul generat normal "
        "cu textul generat după dezactivarea acelui head."
    )

    st.write(
        "Pentru rezultate mai vizibile, folosește prompturi incomplete, de exemplu: "
        "`Banca a aprobat`, `Eu citesc o`, sau "
        "`Pisica tigrată a sărit peste gard, iar după câteva minute ea`."
    )

    col1, col2 = st.columns(2)

    with col1:
        mask_layer = st.slider(
            "Layer to deactivate",
            0,
            num_layers - 1,
            0,
            key="mask_layer"
        )

    with col2:
        mask_head = st.slider(
            "Head to deactivate",
            0,
            num_heads - 1,
            0,
            key="mask_head"
        )

    if st.button("Compare baseline vs masked head"):
        baseline_output = generate_text(prompt)

        masked_output = generate_text_with_head_mask(
            prompt,
            mask_layer,
            mask_head
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline output")
            st.write(baseline_output)

        with col2:
            st.subheader(f"Output without Layer {mask_layer}, Head {mask_head}")
            st.write(masked_output)

        if baseline_output == masked_output:
            st.info(
                "Outputurile sunt identice sau foarte asemănătoare. "
                "Asta poate indica faptul că informația nu depinde doar de acest head, "
                "ci este distribuită în mai multe head-uri."
            )
        else:
            st.warning(
                "Outputurile sunt diferite. Acest lucru poate indica faptul că head-ul dezactivat "
                "influențează predicția modelului pentru acest prompt."
            )





            # ---------------------------------------------------------
    # ROMANIAN VS ENGLISH COMPARISON
    # ---------------------------------------------------------

    st.subheader("Romanian vs English Comparison")

    st.write(
        "Această secțiune compară cum același model procesează propoziții similare "
        "în română și engleză. Comparăm tokenizarea și head-urile care acordă atenție "
        "unei relații alese."
    )

    st.write(
        "Exemplu: în română analizăm relația `ea → Pisica`, iar în engleză relația `it → cat`."
    )

    ro_text = st.text_area(
        "Romanian sentence",
        "Pisica tigrată a sărit peste gard, iar după câteva minute ea s-a întors.",
        key="ro_text"
    )

    en_text = st.text_area(
        "English sentence",
        "The striped cat jumped over the fence, and after a few minutes it returned.",
        key="en_text"
    )

    col1, col2 = st.columns(2)

    with col1:
        ro_source = st.text_input(
            "Romanian source word",
            "ea",
            key="ro_source"
        )

        ro_target = st.text_input(
            "Romanian target word",
            "Pisica",
            key="ro_target"
        )

    with col2:
        en_source = st.text_input(
            "English source word",
            "it",
            key="en_source"
        )

        en_target = st.text_input(
            "English target word",
            "cat",
            key="en_target"
        )

    if st.button("Compare Romanian vs English"):
        ro_result, en_result, summary_df = compare_ro_en(
            ro_text,
            en_text,
            ro_source,
            ro_target,
            en_source,
            en_target
        )

        st.subheader("Tokenization comparison")
        st.dataframe(summary_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Romanian tokens")
            st.write(list(enumerate(ro_result["tokens"])))

            st.subheader("Romanian generated text")
            st.write(ro_result["generated_text"])

        with col2:
            st.subheader("English tokens")
            st.write(list(enumerate(en_result["tokens"])))

            st.subheader("English generated text")
            st.write(en_result["generated_text"])

        st.subheader("Romanian head ranking")

        if ro_result["ranking"] is not None:
            st.write(
                f"Relație analizată: **{ro_source} → {ro_target}**"
            )
            st.dataframe(ro_result["ranking"], use_container_width=True)
        else:
            st.warning(
                "Nu am găsit automat tokenii pentru relația românească. "
                "Încearcă să scrii doar o parte din cuvânt, de exemplu `P` în loc de `Pisica`."
            )

        st.subheader("English head ranking")

        if en_result["ranking"] is not None:
            st.write(
                f"Relation analyzed: **{en_source} → {en_target}**"
            )
            st.dataframe(en_result["ranking"], use_container_width=True)
        else:
            st.warning(
                "Nu am găsit automat tokenii pentru relația engleză. "
                "Încearcă să scrii doar o parte din cuvânt, de exemplu `cat`."
            )

else:
    st.info("Enter a Romanian text and click Generate + Analyze.")