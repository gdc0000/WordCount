import re
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import pandas as pd
import streamlit as st


MAX_NGRAM_SIZE = 5
TOKEN_CLEAN_RE = re.compile(r"[^\w\s']")
TRIE_TERMINAL = "_categories_"


def _tokenize_document(document: str) -> List[str]:
    clean_doc = TOKEN_CLEAN_RE.sub(" ", str(document).lower())
    return clean_doc.split()


def _generate_ngrams(tokens: List[str], lengths: List[int]) -> Counter:
    ngram_counter: Counter = Counter()
    n_tokens = len(tokens)

    for n in lengths:
        if n > n_tokens:
            continue
        ngram_counter.update(" ".join(tokens[i : i + n]) for i in range(n_tokens - n + 1))

    return ngram_counter


def clean_and_tokenize(document: str, max_n: int = MAX_NGRAM_SIZE) -> Tuple[List[str], List[str]]:
    """
    Clean and tokenize a document, generating unigrams and n-grams.
    """
    tokens = _tokenize_document(document)
    ngram_lengths = list(range(2, max_n + 1))
    ngrams = list(_generate_ngrams(tokens, ngram_lengths).elements())
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

    token_set = set(tokens)
    exact_matches = exact_single_words.intersection(token_set)
    detected_words.update(exact_matches)
    count += len(exact_matches)

    token_counter = Counter(tokens)
    for prefix in wildcard_single_prefixes:
        matches = [token for token in token_counter if token.startswith(prefix)]
        detected_words.update(matches)
        count += sum(token_counter[token] for token in matches)

    ngram_set = set(ngrams)
    exact_multi_matches = exact_multi_words.intersection(ngram_set)
    detected_words.update(exact_multi_matches)
    count += len(exact_multi_matches)

    ngram_counter = Counter(ngrams)
    for prefix in wildcard_multi_prefixes:
        matches = [ngram for ngram in ngram_counter if ngram.startswith(prefix)]
        detected_words.update(matches)
        count += sum(ngram_counter[ngram] for ngram in matches)

    return count, list(detected_words)


def _build_prefix_trie(prefix_to_categories: Dict[str, List[str]]) -> Dict[str, dict]:
    trie: Dict[str, dict] = {}

    for prefix, categories in prefix_to_categories.items():
        node = trie
        for char in prefix:
            node = node.setdefault(char, {})
        node.setdefault(TRIE_TERMINAL, []).extend(categories)

    return trie


def _match_prefix_categories(term: str, trie: Dict[str, dict]) -> List[str]:
    if not trie:
        return []

    node = trie
    matched_categories: List[str] = []

    for char in term:
        node = node.get(char)
        if node is None:
            break
        matched_categories.extend(node.get(TRIE_TERMINAL, ()))

    return matched_categories


def _prepare_analysis_config(
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
) -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, dict],
    Dict[str, dict],
    List[int],
]:
    categories = list(exact_single_words.keys())
    exact_single_lookup: DefaultDict[str, List[str]] = defaultdict(list)
    exact_multi_lookup: DefaultDict[str, List[str]] = defaultdict(list)
    wildcard_single_lookup: DefaultDict[str, List[str]] = defaultdict(list)
    wildcard_multi_lookup: DefaultDict[str, List[str]] = defaultdict(list)
    required_ngram_lengths: Set[int] = set()

    for category in categories:
        for term in exact_single_words[category]:
            exact_single_lookup[term].append(category)

        for term in exact_multi_words[category]:
            exact_multi_lookup[term].append(category)
            term_length = len(term.split())
            if 2 <= term_length <= MAX_NGRAM_SIZE:
                required_ngram_lengths.add(term_length)

        for prefix in wildcard_single_prefixes[category]:
            if prefix:
                wildcard_single_lookup[prefix].append(category)

        for prefix in wildcard_multi_prefixes[category]:
            if not prefix:
                continue

            wildcard_multi_lookup[prefix].append(category)
            prefix_length = len(prefix.split())
            if prefix_length <= MAX_NGRAM_SIZE:
                required_ngram_lengths.update(range(max(2, prefix_length), MAX_NGRAM_SIZE + 1))

    return (
        categories,
        dict(exact_single_lookup),
        dict(exact_multi_lookup),
        _build_prefix_trie(dict(wildcard_single_lookup)),
        _build_prefix_trie(dict(wildcard_multi_lookup)),
        sorted(required_ngram_lengths),
    )


def _analyze_document(
    tokens: List[str],
    categories: List[str],
    exact_single_lookup: Dict[str, List[str]],
    exact_multi_lookup: Dict[str, List[str]],
    wildcard_single_trie: Dict[str, dict],
    wildcard_multi_trie: Dict[str, dict],
    required_ngram_lengths: List[int],
) -> Tuple[int, int, Dict[str, int], Dict[str, Set[str]]]:
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    category_counts = {category: 0 for category in categories}
    category_detected = {category: set() for category in categories}

    if tokens:
        token_counter = Counter(tokens)

        for token in token_counter:
            for category in exact_single_lookup.get(token, ()):
                category_counts[category] += 1
                category_detected[category].add(token)

        if wildcard_single_trie:
            for token, occurrences in token_counter.items():
                for category in _match_prefix_categories(token, wildcard_single_trie):
                    category_counts[category] += occurrences
                    category_detected[category].add(token)

    if required_ngram_lengths and n_tokens >= 2:
        ngram_counter = _generate_ngrams(tokens, required_ngram_lengths)

        for ngram in ngram_counter:
            for category in exact_multi_lookup.get(ngram, ()):
                category_counts[category] += 1
                category_detected[category].add(ngram)

        if wildcard_multi_trie:
            for ngram, occurrences in ngram_counter.items():
                for category in _match_prefix_categories(ngram, wildcard_multi_trie):
                    category_counts[category] += occurrences
                    category_detected[category].add(ngram)

    return n_tokens, n_types, category_counts, category_detected


def _analyze_text_core(
    documents: List[str],
    exact_single_words: Dict[str, set],
    wildcard_single_prefixes: Dict[str, List[str]],
    exact_multi_words: Dict[str, set],
    wildcard_multi_prefixes: Dict[str, List[str]],
    progress_bar,
) -> pd.DataFrame:
    (
        categories,
        exact_single_lookup,
        exact_multi_lookup,
        wildcard_single_trie,
        wildcard_multi_trie,
        required_ngram_lengths,
    ) = _prepare_analysis_config(
        exact_single_words,
        wildcard_single_prefixes,
        exact_multi_words,
        wildcard_multi_prefixes,
    )

    n_tokens_list = []
    n_types_list = []
    analysis_results = {
        category: {"word_count": [], "word_perc": [], "detected_words": []}
        for category in categories
    }

    total_docs = len(documents)

    for i, doc in enumerate(documents):
        if pd.isna(doc):
            doc = ""

        tokens = _tokenize_document(doc)
        n_tokens, n_types, category_counts, category_detected = _analyze_document(
            tokens,
            categories,
            exact_single_lookup,
            exact_multi_lookup,
            wildcard_single_trie,
            wildcard_multi_trie,
            required_ngram_lengths,
        )

        n_tokens_list.append(n_tokens)
        n_types_list.append(n_types)

        for category in categories:
            count = category_counts[category]
            word_perc = count / n_tokens if n_tokens > 0 else 0.0
            analysis_results[category]["word_count"].append(count)
            analysis_results[category]["word_perc"].append(word_perc)
            analysis_results[category]["detected_words"].append(list(category_detected[category]))

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
