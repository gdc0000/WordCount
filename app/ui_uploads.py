import streamlit as st


def render_upload_section():
    st.sidebar.header("📥 Upload Files")
    uploaded_dataset = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"]
    )
    uploaded_wordlist = st.sidebar.file_uploader(
        "Upload your wordlist (CSV, TXT, DIC, DICX, or Excel)",
        type=["csv", "txt", "dic", "dicx", "xls", "xlsx"],
    )
    return uploaded_dataset, uploaded_wordlist
