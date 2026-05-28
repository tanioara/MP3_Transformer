import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from generation import generate_text, generate_text_with_head_mask, generate_text_with_multiple_heads_masked
from attention_extractor import extract_attentions
from language_comparison import compare_ro_en
from tokenizer_utils import format_tokens_for_display
import model_loader
from config import AVAILABLE_MODELS, DEFAULT_MODEL


# ---------------------------------------------------------
# GPT-2 BYTE-LEVEL BPE DECODER
# Fixes garbled Romanian chars (ș ț ă â î) that appear as
# weird Latin lookalikes because GPT-2 maps raw bytes to
# Unicode codepoints rather than keeping the original chars.
# ---------------------------------------------------------

def _build_gpt2_unicode_to_byte():
    """Reverse of GPT-2's bytes_to_unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_GPT2_UNICODE_TO_BYTE = _build_gpt2_unicode_to_byte()


def decode_gpt2_token(token: str) -> str:
    """
    Convert a raw GPT-2 token string to a human-readable string.
    Replaces the Ġ space-prefix marker and decodes byte-mapped chars
    back to UTF-8 (fixing Romanian diacritics, etc.).
    """
    # Ġ (U+0120) = leading space in GPT-2 vocab
    token = token.replace("Ġ", " ")
    # Ċ (U+010A) = newline
    token = token.replace("Ċ", "\n")

    # Decode remaining characters through the byte map
    raw_bytes = []
    for ch in token:
        if ch in _GPT2_UNICODE_TO_BYTE:
            raw_bytes.append(_GPT2_UNICODE_TO_BYTE[ch])
        else:
            # Already a normal Unicode char — encode and re-add
            raw_bytes.extend(ch.encode("utf-8"))

    try:
        decoded = bytes(raw_bytes).decode("utf-8")
    except UnicodeDecodeError:
        decoded = bytes(raw_bytes).decode("utf-8", errors="replace")

    return decoded


def clean_tokens(tokens: list[str]) -> list[str]:
    """Apply decode_gpt2_token to a list of raw tokens."""
    return [decode_gpt2_token(t) for t in tokens]


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

st.title("RoGPT2 Attention Analyzer")
st.write(
    "Aplicație pentru analiza attention heads în modele GPT. "
    "Poți introduce un text în română sau engleză, vizualiza atenția dintre tokeni "
    "și testa ce se întâmplă când dezactivezi anumite head-uri. "
    "Compară comportament între modele diferite!"
)

# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

st.divider()
st.subheader("Model Selection")

col1, col2 = st.columns([2, 1])
with col1:
    model_options = list(AVAILABLE_MODELS.keys())
    selected_model = st.selectbox(
        "Choose a model to analyze:",
        model_options,
        index=model_options.index(DEFAULT_MODEL),
        format_func=lambda x: f"{x} - {AVAILABLE_MODELS[x]['description']}"
    )
with col2:
    st.write("")
    if st.button("Load Model", use_container_width=True):
        with st.spinner(f"Loading {selected_model}..."):
            model_loader.load_model(selected_model)
            for key in ["result", "tokens", "attentions", "prompt"]:
                st.session_state.pop(key, None)
        st.success(f"{selected_model} loaded!")
        st.rerun()

st.divider()

# ---------------------------------------------------------
# INPUT TEXT
# ---------------------------------------------------------

st.subheader("Input Text")

prompt = st.text_area("Enter Romanian text:", "Banca a aprobat creditul.", height=150)

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    analyze_clicked = st.button("Generate + Analyze", type="primary", use_container_width=True)
with col_btn2:
    if st.button("Clear results", use_container_width=True):
        for key in ["result", "tokens", "attentions", "prompt"]:
            st.session_state.pop(key, None)
        st.rerun()

if analyze_clicked:
    try:
        with st.spinner("Generating and extracting attentions..."):
            result = generate_text(prompt)
            tokens, attentions = extract_attentions(prompt)
        st.session_state["result"] = result
        st.session_state["tokens"] = tokens
        st.session_state["attentions"] = attentions
        st.session_state["prompt"] = prompt
    except Exception as e:
        st.error(f"Error during generation or attention extraction: {e}")


if "attentions" in st.session_state:

    result     = st.session_state["result"]
    tokens     = st.session_state["tokens"]
    attentions = st.session_state["attentions"]
    prompt     = st.session_state["prompt"]

    num_layers = len(attentions)
    num_heads  = attentions[0].shape[1]
    seq_length = attentions[0].shape[2]

    # Clean tokens once; use everywhere
    tokens_for_attention = tokens[:seq_length]
    fmt_tokens = clean_tokens(tokens_for_attention)

    # ---------------------------------------------------------
    # GENERATED TEXT
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Generated Text")
    st.write(result)

    # ---------------------------------------------------------
    # MODEL STATS
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Model info")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Layers", num_layers)
    m2.metric("Heads per layer", num_heads)
    m3.metric("Sequence length", seq_length)
    m4.metric("Total tokens", len(tokens))

    if len(tokens_for_attention) < len(tokens):
        st.warning(
            f"Tokenizer produced {len(tokens)} tokens but the attention matrix has {seq_length}. "
            f"Using first {seq_length} tokens."
        )

    # ---------------------------------------------------------
    # TOKENS
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Tokens")

    with st.expander("Why are words split into pieces?", expanded=False):
        st.write(
            "GPT-2 uses byte-level BPE tokenization. Every word is split into subword pieces "
            "that were frequent in the training data. Romanian words like *Pisica* may become "
            "`P` + `isi` + `ca` because the full word wasn't common enough. "
            "Characters like `ș`, `ă`, `î` are encoded as raw UTF-8 bytes, which GPT-2 maps "
            "to placeholder Unicode characters — this app decodes them back to the original letters."
        )

    tokens_df = pd.DataFrame({"Index": range(len(fmt_tokens)), "Token": fmt_tokens})
    st.dataframe(tokens_df, use_container_width=True, hide_index=True, height=220)

    # ---------------------------------------------------------
    # ATTENTION EXPLORER
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Attention Explorer")

    col1, col2 = st.columns(2)
    with col1:
        layer = st.slider("Layer", 0, num_layers - 1, 0, key="explorer_layer")
    with col2:
        head = st.slider("Head", 0, num_heads - 1, 0, key="explorer_head")

    token_index = st.selectbox(
        "Select token to inspect",
        list(range(len(fmt_tokens))),
        format_func=lambda i: f"{i}: {fmt_tokens[i]}"
    )

    matrix = attentions[layer][0, head].detach().cpu().numpy()

    st.subheader("Attention Heatmap")
    st.caption("Each row shows how much attention that token pays to every other token. Darker = more attention.")

    fig = px.imshow(
        matrix,
        x=fmt_tokens,
        y=fmt_tokens,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Token attended to", y="Current token", color="Attention"),
        title=f"Layer {layer} · Head {head}"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TOP ATTENDED TOKENS
    # ---------------------------------------------------------

    scores  = matrix[token_index]
    top_idx = np.argsort(scores)[::-1][:5]

    st.subheader(f"Top attended tokens for '{fmt_tokens[token_index]}'")
    st.caption("Tokens this token pays the most attention to in the selected layer / head.")

    top_df = pd.DataFrame({
        "Rank":       list(range(1, len(top_idx) + 1)),
        "Token":      [fmt_tokens[i] for i in top_idx],
        "Score":      [round(float(scores[i]), 4) for i in top_idx],
        "Percentage": [f"{float(scores[i]) * 100:.1f}%" for i in top_idx],
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # HEAD RELEVANCE RANKING  (multi-token source & target)
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Head relevance ranking")
    st.write(
        "Select one or more tokens as source and target. "
        "Useful for Romanian words split across multiple subword tokens — "
        "e.g. select `P`, `isi`, `ca` together to represent *Pisica*."
    )

    token_options = list(range(len(fmt_tokens)))
    token_labels  = [f"{i}: {fmt_tokens[i]}" for i in token_options]

    col1, col2 = st.columns(2)
    with col1:
        source_indices = st.multiselect(
            "Source tokens (doing the attending)",
            token_options,
            default=[0],
            format_func=lambda i: token_labels[i],
            key="source_tokens_multi"
        )
    with col2:
        target_indices = st.multiselect(
            "Target tokens (being attended to)",
            token_options,
            default=[min(1, len(fmt_tokens) - 1)],
            format_func=lambda i: token_labels[i],
            key="target_tokens_multi"
        )

    if not source_indices or not target_indices:
        st.warning("Select at least one source token and one target token.")
    else:
        source_label = " + ".join(fmt_tokens[i] for i in source_indices)
        target_label = " + ".join(fmt_tokens[i] for i in target_indices)
        rel_label    = f"[{source_label}] → [{target_label}]"

        ranking = []
        for l in range(num_layers):
            for h in range(num_heads):
                attn_matrix = attentions[l][0, h].detach().cpu().numpy()

                # Average attention across all (source, target) token pairs
                pair_scores = [
                    float(attn_matrix[s, t])
                    for s in source_indices
                    for t in target_indices
                ]
                direct_score   = float(np.mean(pair_scores))
                avg_from_source = float(np.mean([attn_matrix[s].mean() for s in source_indices]))

                # Average entropy across source tokens
                entropies = []
                for s in source_indices:
                    dist = np.clip(attn_matrix[s], 1e-10, 1.0)
                    entropies.append(-np.sum(dist * np.log(dist)))
                entropy = float(np.mean(entropies))

                ranking.append({
                    "Layer":        l,
                    "Head":         h,
                    "Direct Score": round(direct_score, 4),
                    "Avg Attention": round(avg_from_source, 4),
                    "Entropy":      round(entropy, 4),
                })

        ranking_df = (
            pd.DataFrame(ranking)
            .sort_values("Direct Score", ascending=False)
            .reset_index(drop=True)
        )

        st.write(f"Relation: **{rel_label}**")

        tab_table, tab_chart = st.tabs(["Table", "Chart"])
        with tab_table:
            st.dataframe(ranking_df.head(10), use_container_width=True)
        with tab_chart:
            top10 = ranking_df.head(10).copy()
            top10["Layer-Head"] = top10.apply(
                lambda r: f"L{int(r['Layer'])}H{int(r['Head'])}", axis=1
            )
            bar_fig = px.bar(
                top10,
                x="Layer-Head",
                y="Direct Score",
                title=f"Top 10 heads: {rel_label}",
                labels={"Direct Score": "Attention score", "Layer-Head": "Head"},
                color="Direct Score",
                color_continuous_scale="Blues"
            )
            bar_fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(bar_fig, use_container_width=True)

    # ---------------------------------------------------------
    # HEAD ABLATION EXPERIMENT
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Head Ablation Analysis")
    st.write(
        "Compare generated text with and without specific heads. "
        "Use temperature > 0 to see clearer differences."
    )

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=2.0, value=0.8, step=0.1,
            help="0 = deterministic, 1 = normal, >1 = very random"
        )
    with col2:
        top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0, max_value=1.0, value=0.9, step=0.05
        )

    ablation_tab1, ablation_tab2 = st.tabs(["Single Head", "Multiple Heads"])

    with ablation_tab1:
        st.caption("Incomplete prompts (e.g. `Banca a aprobat`) show bigger differences than complete sentences.")

        col1, col2, col3 = st.columns(3)
        with col1:
            ablation_layer = st.slider("Layer", 0, num_layers - 1, 0, key="ablation_layer")
        with col2:
            ablation_head = st.slider("Head", 0, num_heads - 1, 0, key="ablation_head")
        with col3:
            st.write("")
            generate_btn = st.button("Generate & Compare", key="analyze_head", use_container_width=True, type="primary")

        if generate_btn:
            try:
                with st.spinner("Generating..."):
                    baseline_output = generate_text(prompt, temperature=temperature, top_p=top_p)
                    ablated_output  = generate_text_with_head_mask(
                        prompt, ablation_layer, ablation_head,
                        temperature=temperature, top_p=top_p
                    )
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Baseline**")
                    st.code(baseline_output, language="text")
                with col2:
                    st.write(f"**Without L{ablation_layer} H{ablation_head}**")
                    st.code(ablated_output, language="text")
                if baseline_output == ablated_output:
                    st.info("Outputs are identical — try higher temperature or a different head.")
                else:
                    st.success("Outputs differ — this head affects generation.")
            except Exception as e:
                st.error(f"Error during ablation: {e}")

    with ablation_tab2:
        st.caption("Disable multiple heads at once to see compounding effects.")

        num_selections = st.slider("Number of heads to ablate", 1, 5, 2, key="num_heads_ablate")
        heads_to_ablate = []
        cols = st.columns(min(3, num_selections))
        for i in range(num_selections):
            with cols[i % len(cols)]:
                st.write(f"**Head {i + 1}**")
                c1, c2 = st.columns(2)
                with c1:
                    l_val = st.number_input(f"Layer #{i+1}", 0, num_layers - 1, 0, key=f"mlayer_{i}")
                with c2:
                    h_val = st.number_input(f"Head #{i+1}", 0, num_heads - 1, 0, key=f"mhead_{i}")
                heads_to_ablate.append((l_val, h_val))

        if st.button("Generate & Compare", key="analyze_multi_head", use_container_width=True, type="primary"):
            try:
                with st.spinner(f"Generating with {len(heads_to_ablate)} heads ablated..."):
                    baseline_output     = generate_text(prompt, temperature=temperature, top_p=top_p)
                    multi_ablated_output = generate_text_with_multiple_heads_masked(
                        prompt, heads_to_ablate, temperature=temperature, top_p=top_p
                    )
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Baseline**")
                    st.code(baseline_output, language="text")
                with col2:
                    heads_str = ", ".join([f"L{l}H{h}" for l, h in heads_to_ablate])
                    st.write(f"**Without: {heads_str}**")
                    st.code(multi_ablated_output, language="text")
                if baseline_output == multi_ablated_output:
                    st.info("Outputs are identical — these heads may be redundant.")
                else:
                    st.success("Outputs differ — these heads affect generation.")
            except Exception as e:
                st.error(f"Error during multi-head ablation: {e}")

    # ---------------------------------------------------------
    # ROMANIAN VS ENGLISH COMPARISON
    # ---------------------------------------------------------

    st.divider()
    st.subheader("Romanian vs English Comparison")
    st.write(
        "Tokenize both sentences first, then pick source and target tokens by index "
        "for each language. Romanian words are often split across multiple tokens — "
        "select all pieces that belong to the same word."
    )

    col1, col2 = st.columns(2)
    with col1:
        ro_text = st.text_area(
            "Romanian sentence",
            "Pisica tigrata a sarit peste gard, iar dupa cateva minute ea s-a intors.",
            key="ro_text", height=120
        )
    with col2:
        en_text = st.text_area(
            "English sentence",
            "The striped cat jumped over the fence, and after a few minutes it returned.",
            key="en_text", height=120
        )

    if st.button("Tokenize both sentences", use_container_width=True):
        try:
            with st.spinner("Tokenizing..."):
                ro_tokens, ro_attns = extract_attentions(ro_text)
                en_tokens, en_attns = extract_attentions(en_text)
            st.session_state["ro_cmp_tokens"] = ro_tokens
            st.session_state["ro_cmp_attns"]  = ro_attns
            st.session_state["en_cmp_tokens"] = en_tokens
            st.session_state["en_cmp_attns"]  = en_attns
        except Exception as e:
            st.error(f"Error during tokenization: {e}")

    if "ro_cmp_tokens" in st.session_state:
        ro_cmp_tokens = st.session_state["ro_cmp_tokens"]
        ro_cmp_attns  = st.session_state["ro_cmp_attns"]
        en_cmp_tokens = st.session_state["en_cmp_tokens"]
        en_cmp_attns  = st.session_state["en_cmp_attns"]

        ro_cmp_seq = ro_cmp_attns[0].shape[2]
        en_cmp_seq = en_cmp_attns[0].shape[2]

        ro_fmt = clean_tokens(ro_cmp_tokens[:ro_cmp_seq])
        en_fmt = clean_tokens(en_cmp_tokens[:en_cmp_seq])

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Romanian tokens")
            st.dataframe(
                pd.DataFrame({"Index": range(len(ro_fmt)), "Token": ro_fmt}),
                use_container_width=True, hide_index=True, height=200
            )
        with col2:
            st.caption("English tokens")
            st.dataframe(
                pd.DataFrame({"Index": range(len(en_fmt)), "Token": en_fmt}),
                use_container_width=True, hide_index=True, height=200
            )

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Romanian relation**")
            ro_src_idx = st.multiselect(
                "Source tokens",
                list(range(len(ro_fmt))),
                default=[0],
                format_func=lambda i: f"{i}: {ro_fmt[i]}",
                key="ro_cmp_src"
            )
            ro_tgt_idx = st.multiselect(
                "Target tokens",
                list(range(len(ro_fmt))),
                default=[min(1, len(ro_fmt) - 1)],
                format_func=lambda i: f"{i}: {ro_fmt[i]}",
                key="ro_cmp_tgt"
            )
        with col2:
            st.write("**English relation**")
            en_src_idx = st.multiselect(
                "Source tokens",
                list(range(len(en_fmt))),
                default=[0],
                format_func=lambda i: f"{i}: {en_fmt[i]}",
                key="en_cmp_src"
            )
            en_tgt_idx = st.multiselect(
                "Target tokens",
                list(range(len(en_fmt))),
                default=[min(1, len(en_fmt) - 1)],
                format_func=lambda i: f"{i}: {en_fmt[i]}",
                key="en_cmp_tgt"
            )

        if st.button("Compare head rankings", type="primary", use_container_width=True):
            if not ro_src_idx or not ro_tgt_idx:
                st.warning("Select at least one source and one target token for Romanian.")
            elif not en_src_idx or not en_tgt_idx:
                st.warning("Select at least one source and one target token for English.")
            else:
                def compute_cmp_ranking(attns, src_indices, tgt_indices):
                    rows = []
                    for l in range(len(attns)):
                        for h in range(attns[0].shape[1]):
                            m = attns[l][0, h].detach().cpu().numpy()
                            direct = float(np.mean([m[s, t] for s in src_indices for t in tgt_indices]))
                            avg    = float(np.mean([m[s].mean() for s in src_indices]))
                            ent    = float(np.mean([
                                -np.sum(np.clip(m[s], 1e-10, 1.0) * np.log(np.clip(m[s], 1e-10, 1.0)))
                                for s in src_indices
                            ]))
                            rows.append({"Layer": l, "Head": h,
                                         "Direct Score": round(direct, 4),
                                         "Avg Attention": round(avg, 4),
                                         "Entropy": round(ent, 4)})
                    return (pd.DataFrame(rows)
                              .sort_values("Direct Score", ascending=False)
                              .reset_index(drop=True))

                ro_rank_df = compute_cmp_ranking(ro_cmp_attns, ro_src_idx, ro_tgt_idx)
                en_rank_df = compute_cmp_ranking(en_cmp_attns, en_src_idx, en_tgt_idx)

                ro_rel = "[" + " + ".join(ro_fmt[i] for i in ro_src_idx) + "] -> [" + " + ".join(ro_fmt[i] for i in ro_tgt_idx) + "]"
                en_rel = "[" + " + ".join(en_fmt[i] for i in en_src_idx) + "] -> [" + " + ".join(en_fmt[i] for i in en_tgt_idx) + "]"

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Romanian:** {ro_rel}")
                    tab1, tab2 = st.tabs(["Table", "Chart"])
                    with tab1:
                        st.dataframe(ro_rank_df.head(10), use_container_width=True)
                    with tab2:
                        t = ro_rank_df.head(10).copy()
                        t["Layer-Head"] = t.apply(lambda r: f"L{int(r['Layer'])}H{int(r['Head'])}", axis=1)
                        st.plotly_chart(
                            px.bar(t, x="Layer-Head", y="Direct Score", title=ro_rel,
                                   color="Direct Score", color_continuous_scale="Blues")
                              .update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0)),
                            use_container_width=True
                        )
                with col2:
                    st.write(f"**English:** {en_rel}")
                    tab1, tab2 = st.tabs(["Table", "Chart"])
                    with tab1:
                        st.dataframe(en_rank_df.head(10), use_container_width=True)
                    with tab2:
                        t = en_rank_df.head(10).copy()
                        t["Layer-Head"] = t.apply(lambda r: f"L{int(r['Layer'])}H{int(r['Head'])}", axis=1)
                        st.plotly_chart(
                            px.bar(t, x="Layer-Head", y="Direct Score", title=en_rel,
                                   color="Direct Score", color_continuous_scale="Oranges")
                              .update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0)),
                            use_container_width=True
                        )

    st.info("Enter a Romanian text above and click Generate + Analyze.")