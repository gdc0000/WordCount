import streamlit as st


def render_category_selector(categories):
    return st.multiselect(
        "📌 Select Categories to Analyze", options=categories, default=categories
    )


def render_text_column_selector(dataset):
    text_columns = dataset.select_dtypes(include=["object", "string"]).columns.tolist()
    if not text_columns:
        st.error("❌ No text columns found in the dataset. Please upload a dataset with at least one text column.")
        return None
    return st.selectbox("🔍 Select the column containing text data:", text_columns)


def render_start_analysis_button():
    return st.button("🚀 Start Analysis")
