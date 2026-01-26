import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple


@st.cache_data
def load_dataset(uploaded_file):
    """
    Load a dataset from a CSV, TSV, or Excel file.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith(".tsv"):
            return pd.read_csv(uploaded_file, sep="\t")
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        st.error("Unsupported file format for dataset. Please upload a CSV, TSV, or Excel file.")
        return None
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        return None


@st.cache_data
def load_wordlist(
    uploaded_file,
) -> Tuple[Dict[str, set], Dict[str, List[str]], Dict[str, set], Dict[str, List[str]]]:
    """
    Load and preprocess a multi-category wordlist.
    """
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            wordlist_df = pd.read_csv(uploaded_file)
        elif file_extension in ["txt", "dic"]:
            wordlist_df = pd.read_csv(uploaded_file, sep="\t")
        elif file_extension == "dicx":
            wordlist_df = pd.read_csv(uploaded_file, sep=",")
        elif file_extension in ["xls", "xlsx"]:
            wordlist_df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format for wordlist. Please upload a CSV, TXT, DIC, DICX, or Excel file.")
            return None, None, None, None

        if "DicTerm" not in wordlist_df.columns:
            st.error("The wordlist file must contain a column named 'DicTerm'.")
            return None, None, None, None

        category_columns = [col for col in wordlist_df.columns if col != "DicTerm"]
        if not category_columns:
            st.error("No category columns found in the wordlist file.")
            return None, None, None, None

        sanitized_category_columns = [col.replace(" ", "_") for col in category_columns]
        wordlist_df.columns = ["DicTerm"] + sanitized_category_columns

        exact_single_words = {category: set() for category in sanitized_category_columns}
        wildcard_single_prefixes = {category: [] for category in sanitized_category_columns}
        exact_multi_words = {category: set() for category in sanitized_category_columns}
        wildcard_multi_prefixes = {category: [] for category in sanitized_category_columns}

        for _, row in wordlist_df.iterrows():
            term = str(row["DicTerm"]).strip().lower()
            is_multi_word = len(term.split()) > 1
            for category in sanitized_category_columns:
                cell_value = row[category]
                if pd.notna(cell_value) and str(cell_value).strip().upper() == "X":
                    if term.endswith("*"):
                        prefix = term[:-1].strip()
                        if prefix:
                            if is_multi_word:
                                wildcard_multi_prefixes[category].append(prefix)
                            else:
                                wildcard_single_prefixes[category].append(prefix)
                    else:
                        if is_multi_word:
                            exact_multi_words[category].add(term)
                        else:
                            exact_single_words[category].add(term)

        return exact_single_words, wildcard_single_prefixes, exact_multi_words, wildcard_multi_prefixes
    except Exception as exc:
        st.error(f"Error loading wordlist: {exc}")
        return None, None, None, None
