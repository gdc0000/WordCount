import streamlit as st

from .summary import generate_summary_list


def render_dataset_preview(dataset) -> None:
    st.subheader("?? Dataset Preview")
    st.dataframe(dataset.head())


def render_wordlist_summary(exact_single_words, exact_multi_words) -> None:
    st.subheader("?? Wordlist Summary")
    summary_text = generate_summary_list(exact_single_words, exact_multi_words)
    st.markdown(summary_text)


def render_wordlists_summary(summaries) -> None:
    st.subheader("?? Wordlist Summary")
    for label, exact_single, exact_multi in summaries:
        st.markdown(f"**{label}**")
        summary_text = generate_summary_list(exact_single, exact_multi)
        st.markdown(summary_text)
