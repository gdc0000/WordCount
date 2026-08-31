"""Tests for ``wordcount.core.tokenize`` — the pure tokenization primitives.

Covers: whitespace/case/punct handling, apostrophe retention, n-gram lengths,
``max_n`` honored end-to-end, and that ``generate_ngrams`` returns a ``Counter``
that counts duplicate occurrences (the basis for the §2.1 single counting
convention implemented in core/counting.py).

Pure: no pandas, no streamlit, no fastapi.
"""

from __future__ import annotations

from collections import Counter

import pytest

from wordcount.core.tokenize import (
    DEFAULT_MAX_N,
    clean_and_tokenize,
    generate_candidate_ngrams,
    generate_ngrams,
    ngram_lengths_for,
    tokenize,
)


# --------------------------------------------------------------------------- #
# tokenize
# --------------------------------------------------------------------------- #
def test_tokenize_lowercases() -> None:
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_strips_punctuation_to_spaces() -> None:
    # Non-[\w\s'] runs become a single space, so punctuation separates tokens.
    assert tokenize("Hello, world!") == ["hello", "world"]
    assert tokenize("well...being") == ["well", "being"]


def test_tokenize_keeps_apostrophes() -> None:
    # Apostrophes are retained so contractions/possessives stay intact.
    assert tokenize("don't stop") == ["don't", "stop"]
    assert tokenize("it's a test") == ["it's", "a", "test"]


def test_tokenize_collapses_whitespace() -> None:
    assert tokenize("  too   much   space  ") == ["too", "much", "space"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_only_punctuation() -> None:
    assert tokenize("!!! ??? ...") == []


def test_tokenize_digits_and_underscores_kept() -> None:
    # \w includes digits and underscore.
    assert tokenize("item_1 item_2") == ["item_1", "item_2"]


# --------------------------------------------------------------------------- #
# generate_ngrams
# --------------------------------------------------------------------------- #
def test_generate_ngrams_returns_counter() -> None:
    ngrams = generate_ngrams(["a", "b", "a", "b"], [2])
    assert isinstance(ngrams, Counter)


def test_generate_ngrams_counts_duplicates() -> None:
    # tokens a b a b -> bigrams "a b", "b a", "a b" => "a b":2, "b a":1
    ngrams = generate_ngrams(["a", "b", "a", "b"], [2])
    assert ngrams["a b"] == 2
    assert ngrams["b a"] == 1


def test_generate_ngrams_multiple_lengths() -> None:
    tokens = ["a", "b", "c"]
    ngrams = generate_ngrams(tokens, [2, 3])
    assert ngrams["a b"] == 1
    assert ngrams["b c"] == 1
    assert ngrams["a b c"] == 1


def test_generate_ngrams_skips_n_greater_than_tokens() -> None:
    # 4-grams from 3 tokens -> nothing added for that order.
    ngrams = generate_ngrams(["a", "b", "c"], [2, 4])
    assert ngrams["a b"] == 1
    assert ngrams["b c"] == 1
    # No 4-gram keys present.
    assert all(" " not in k or k.count(" ") != 3 for k in ngrams)


def test_generate_ngrams_skips_non_positive_lengths() -> None:
    # n <= 0 is rejected; n >= 1 is accepted (the "start at 2" convention is
    # enforced by ngram_lengths_for, not here, so callers can request unigrams).
    ngrams = generate_ngrams(["a", "b"], [0, 1, -1, 2])
    assert ngrams["a b"] == 1
    # n=1 is valid here: unigrams are produced.
    assert ngrams["a"] == 1
    assert ngrams["b"] == 1
    # n=0 / n=-1 contribute nothing — the only keys are the unigrams + bigram.
    assert set(ngrams) == {"a", "b", "a b"}


def test_generate_ngrams_empty_tokens() -> None:
    assert generate_ngrams([], [2, 3]) == Counter()


# --------------------------------------------------------------------------- #
# generate_candidate_ngrams — first-word filtered generation (perf)
# --------------------------------------------------------------------------- #
def test_candidate_ngrams_equals_full_when_all_first_words() -> None:
    """Filter with every possible first word = plain generate_ngrams."""
    tokens = ["a", "b", "a", "b", "c"]
    first = frozenset(tokens)
    lengths = [2, 3]
    assert generate_candidate_ngrams(tokens, lengths, first) == generate_ngrams(tokens, lengths)


def test_candidate_ngrams_skips_windows_with_unknown_first_token() -> None:
    """Windows not starting on a candidate word are never joined."""
    tokens = ["x", "a", "b", "x", "a", "b", "c"]
    # Only windows starting with "a" are generated: index 1 -> "a b", "a b x";
    # index 4 -> "a b", "a b c". Windows starting with "x" are never joined.
    got = generate_candidate_ngrams(tokens, [2, 3], frozenset({"a"}))
    assert got == Counter({"a b": 2, "a b x": 1, "a b c": 1})


def test_candidate_ngrams_empty_first_words_returns_empty() -> None:
    tokens = ["a", "b", "c"]
    assert generate_candidate_ngrams(tokens, [2, 3], frozenset()) == Counter()


def test_candidate_ngrams_counts_duplicates() -> None:
    tokens = ["new", "york", "new", "york"]
    got = generate_candidate_ngrams(tokens, [2], frozenset({"new"}))
    assert got == Counter({"new york": 2})


# --------------------------------------------------------------------------- #
# ngram_lengths_for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("max_n", "expected"),
    [
        (1, []),
        (2, [2]),
        (3, [2, 3]),
        (5, [2, 3, 4, 5]),
        (0, []),
        (-1, []),
    ],
)
def test_ngram_lengths_for(max_n: int, expected: list[int]) -> None:
    assert ngram_lengths_for(max_n) == expected


def test_default_max_n_is_three() -> None:
    # Lowered from the legacy hard-coded 5 (§5.1). Surfaced via API/CLI, not
    # a module-level cap.
    assert DEFAULT_MAX_N == 3


# --------------------------------------------------------------------------- #
# clean_and_tokenize — end-to-end primitive
# --------------------------------------------------------------------------- #
def test_clean_and_tokenize_returns_tokens_and_counter() -> None:
    tokens, ngrams = clean_and_tokenize("the quick brown fox", max_n=3)
    assert tokens == ["the", "quick", "brown", "fox"]
    assert isinstance(ngrams, Counter)
    # bigrams + trigrams for 4 tokens
    assert ngrams["the quick"] == 1
    assert ngrams["quick brown"] == 1
    assert ngrams["the quick brown"] == 1


def test_clean_and_tokenize_max_n_honored() -> None:
    text = "one two three four five"
    _, ngrams_2 = clean_and_tokenize(text, max_n=2)
    _, ngrams_3 = clean_and_tokenize(text, max_n=3)
    # max_n=2 => only bigrams, no trigrams.
    assert "one two three" not in ngrams_2
    assert "one two three" in ngrams_3
    # max_n=2 has strictly fewer distinct phrase keys than max_n=3.
    assert len(ngrams_2) < len(ngrams_3)


def test_clean_and_tokenize_uses_default_max_n() -> None:
    text = "a b c d"
    _, ngrams = clean_and_tokenize(text)  # default max_n=3
    assert "a b" in ngrams
    assert "a b c" in ngrams
    assert "a b c d" not in ngrams  # 4-grams excluded under default


def test_clean_and_tokenize_single_word_has_no_ngrams() -> None:
    tokens, ngrams = clean_and_tokenize("lonely", max_n=3)
    assert tokens == ["lonely"]
    assert ngrams == Counter()
