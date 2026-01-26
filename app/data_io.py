from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from .naming import sanitize_identifier, uniquify


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


def load_wordlists(
    uploaded_files,
    prefixes: Dict[str, str],
) -> Tuple[Dict[str, set], Dict[str, List[str]], Dict[str, set], Dict[str, List[str]], List[Tuple[str, Dict, Dict]]]:
    combined_exact_single: Dict[str, set] = {}
    combined_wildcard_single: Dict[str, List[str]] = {}
    combined_exact_multi: Dict[str, set] = {}
    combined_wildcard_multi: Dict[str, List[str]] = {}
    summaries: List[Tuple[str, Dict, Dict]] = []
    existing = set()

    for uploaded_file in uploaded_files:
        exact_single, wildcard_single, exact_multi, wildcard_multi = load_wordlist(uploaded_file)
        if any(item is None for item in (exact_single, wildcard_single, exact_multi, wildcard_multi)):
            continue

        stem = Path(uploaded_file.name).stem
        prefix = prefixes.get(uploaded_file.name, stem)
        prefix = sanitize_identifier(prefix or stem)

        renamed_single = {}
        renamed_multi = {}

        for category in exact_single.keys():
            safe_category = sanitize_identifier(category)
            combined_name = f"{prefix}_{safe_category}" if prefix else safe_category
            combined_name = sanitize_identifier(combined_name)
            combined_name = uniquify(combined_name, existing)
            existing.add(combined_name)

            combined_exact_single[combined_name] = exact_single[category]
            combined_wildcard_single[combined_name] = wildcard_single[category]
            combined_exact_multi[combined_name] = exact_multi[category]
            combined_wildcard_multi[combined_name] = wildcard_multi[category]

            renamed_single[combined_name] = exact_single[category]
            renamed_multi[combined_name] = exact_multi[category]

        summaries.append((prefix, renamed_single, renamed_multi))

    return (
        combined_exact_single,
        combined_wildcard_single,
        combined_exact_multi,
        combined_wildcard_multi,
        summaries,
    )
