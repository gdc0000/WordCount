import streamlit as st

from .naming import sanitize_identifier


def render_category_selector(categories):
    return st.multiselect(
        "?? Select Categories to Analyze", options=categories, default=categories
    )


def render_text_column_selector(dataset):
    text_columns = dataset.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_columns:
        st.error("? No text columns found in the dataset. Please upload a dataset with at least one text column.")
        return None
    return st.selectbox("?? Select the column containing text data:", text_columns)


def render_start_analysis_button():
    return st.button("?? Start Analysis")


def render_wordlist_prefixes(uploaded_wordlists):
    st.subheader("?? Wordlist Namespaces")
    st.caption("Each wordlist gets a prefix to keep columns unique.")
    prefixes = {}
    for uploaded_file in uploaded_wordlists:
        default_prefix = sanitize_identifier(uploaded_file.name.rsplit(".", 1)[0])
        prefix = st.text_input(
            f"Prefix for {uploaded_file.name}",
            value=default_prefix,
            key=f"prefix_{uploaded_file.name}",
        )
        prefixes[uploaded_file.name] = sanitize_identifier(prefix or default_prefix)
    return prefixes
