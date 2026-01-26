import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


def clean_and_tokenize(document: str, max_n: int = 5) -> Tuple[List[str], List[str]]:
    """
    Clean and tokenize a document, generating unigrams and n-grams.
    """
    clean_doc = re.sub(r"[^\w\s']", " ", document.lower())
    tokens = clean_doc.split()

    ngrams = []
    for n in range(2, max_n + 1):
        n_grams = zip(*[tokens[i:] for i in range(n)])
        ngrams += [" ".join(gram) for gram in n_grams]

    return tokens, ngrams


def count_words(
    tokens: List[str],
    ngrams: List[str],
    exact_single_words: set,
    wildcard_single_prefixes: List[str],
    exact_multi_words: set,
    wildcard_multi_prefixes: List[str],
) -> Tuple[int, List[str]]:
    """
    Count words and n-grams based on exact matches and wildcard prefixes.
    """
    detected_words = set()
    count = 0

    exact_matches = exact_single_words.intersection(tokens)
    detected_words.update(exact_matches)
    count += len(exact_matches)

    for prefix in wildcard_single_prefixes:
        matches = [token for token in tokens if token.startswith(prefix)]
        detected_words.update(matches)
        count += len(matches)

    exact_multi_matches = exact_multi_words.intersection(ngrams)
    detected_words.update(exact_multi_matches)
    count += len(exact_multi_matches)

    for prefix in wildcard_multi_prefixes:
        matches = [ngram for ngram in ngrams if ngram.startswith(prefix)]
        detected_words.update(matches)
        count += len(matches)

    return count, list(detected_words)


def _analyze_text_core(
    documents: List[str],
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
    progress_bar,
) -> pd.DataFrame:
    n_tokens_list = []
    n_types_list = []

    analysis_results = {
        category: {"word_count": [], "word_perc": [], "detected_words": []}
        for category in exact_single_words.keys()
    }

    total_docs = len(documents)

    for i, doc in enumerate(documents):
        if pd.isna(doc):
            doc = ""
        tokens, ngrams = clean_and_tokenize(doc)
        n_tokens = len(tokens)
        n_types = len(set(tokens))

        n_tokens_list.append(n_tokens)
        n_types_list.append(n_types)

        for category in exact_single_words.keys():
            count, detected = count_words(
                tokens,
                ngrams,
                exact_single_words[category],
                wildcard_single_prefixes[category],
                exact_multi_words[category],
                wildcard_multi_prefixes[category],
            )
            word_perc = count / n_tokens if n_tokens > 0 else 0.0
            analysis_results[category]["word_count"].append(count)
            analysis_results[category]["word_perc"].append(word_perc)
            analysis_results[category]["detected_words"].append(detected)

        if progress_bar and ((i + 1) % 200 == 0 or i == total_docs - 1):
            progress_bar.progress((i + 1) / total_docs)

    global_metrics = pd.DataFrame({"n_tokens": n_tokens_list, "n_types": n_types_list})

    category_metrics = {}
    for category, metrics in analysis_results.items():
        category_metrics[f"{category}_word_count"] = metrics["word_count"]
        category_metrics[f"{category}_word_perc"] = metrics["word_perc"]
        category_metrics[f"{category}_detected_words"] = metrics["detected_words"]

    category_metrics_df = pd.DataFrame(category_metrics)
    analysis_df = pd.concat([global_metrics, category_metrics_df], axis=1)
    return analysis_df


def _normalize_wordlists_for_cache(
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
    exact_single_norm = {k: tuple(sorted(v)) for k, v in exact_single_words.items()}
    wildcard_single_norm = {k: tuple(v) for k, v in wildcard_single_prefixes.items()}
    exact_multi_norm = {k: tuple(sorted(v)) for k, v in exact_multi_words.items()}
    wildcard_multi_norm = {k: tuple(v) for k, v in wildcard_multi_prefixes.items()}
    return exact_single_norm, wildcard_single_norm, exact_multi_norm, wildcard_multi_norm


@st.cache_data
def analyze_text_cached(
    documents: List[str],
    exact_single_words_norm: Dict[str, Tuple[str, ...]],
    wildcard_single_prefixes_norm: Dict[str, Tuple[str, ...]],
    exact_multi_words_norm: Dict[str, Tuple[str, ...]],
    wildcard_multi_prefixes_norm: Dict[str, Tuple[str, ...]],
) -> pd.DataFrame:
    exact_single_words = {k: set(v) for k, v in exact_single_words_norm.items()}
    wildcard_single_prefixes = {k: list(v) for k, v in wildcard_single_prefixes_norm.items()}
    exact_multi_words = {k: set(v) for k, v in exact_multi_words_norm.items()}
    wildcard_multi_prefixes = {k: list(v) for k, v in wildcard_multi_prefixes_norm.items()}
    return _analyze_text_core(
        documents,
        exact_single_words,
        wildcard_single_prefixes,
        exact_multi_words,
        wildcard_multi_prefixes,
        progress_bar=None,
    )


def analyze_text(
    documents: List[str],
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
    progress_bar,
) -> pd.DataFrame:
    return _analyze_text_core(
        documents,
        exact_single_words,
        wildcard_single_prefixes,
        exact_multi_words,
        wildcard_multi_prefixes,
        progress_bar,
    )


def normalize_wordlists_for_cache(
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
):
    return _normalize_wordlists_for_cache(
        exact_single_words,
        wildcard_single_prefixes,
        exact_multi_words,
        wildcard_multi_prefixes,
    )
