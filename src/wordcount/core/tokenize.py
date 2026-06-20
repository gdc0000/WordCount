"""Tokenization and n-gram generation.

Pure: no pandas, no streamlit, no fastapi. The tokenization primitives
(``str.lower``, ``re.sub``, ``str.split``) are already C-implemented in CPython,
so there is no per-token Python loop here — the loop lives only over the
*resulting* tokens during matching (core/counting.py). (See the perf sub-audit
evaluation: a HF/Rust tokenizer would not speed this up materially and would
break the matching contract, so it is rejected.)

``max_n`` is always a parameter here; the module-level ``DEFAULT_MAX_N`` is the
single source of truth for the default, surfaced via the API/CLI rather than
hard-coded (fixes §5.1: the old ``MAX_NGRAM_SIZE = 5`` was neither configurable
nor correctly documented). Default lowered to 3: 5-grams quadruple the n-gram
work for near-zero gain in social-science dictionaries (perf evaluation §2).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

DEFAULT_MAX_N: int = 3
"""Default maximum n-gram order. Overridable via API/CLI; not a hard cap."""

#: Multi-word n-gram generation starts at this order (unigrams are the tokens
#: themselves, handled separately).
MIN_NGRAM_LENGTH: int = 2

TOKEN_CLEAN_RE = re.compile(r"[^\w\s']")
"""Regex replacing non-``[\\w\\s']`` runs with a space before splitting."""


def tokenize(document: str) -> list[str]:
    """Lowercase, strip non-``[\\w\\s']`` chars to spaces, split on whitespace.

    ``NaN``/``None`` are coerced via ``str()`` to ``"nan"``/``"none"`` by the
    caller's contract; the analysis loop in core/counting.py guards ``pd.isna``
    before calling this, so this function receives a real string.
    """
    cleaned = TOKEN_CLEAN_RE.sub(" ", document.lower())
    return cleaned.split()


def generate_ngrams(tokens: Sequence[str], lengths: Sequence[int]) -> Counter[str]:
    """Generate n-grams for the given orders, returning a ``Counter``.

    ``Counter`` (not a list) so duplicates collapse to frequencies and the
    matching layer can read occurrence counts directly — the basis for the
    single token-frequency counting convention that fixes §2.1.

    Orders where ``n > len(tokens)`` are skipped silently.
    """
    ngram_counter: Counter[str] = Counter()
    n_tokens = len(tokens)
    for n in lengths:
        if n > n_tokens or n < 1:
            continue
        ngram_counter.update(" ".join(tokens[i : i + n]) for i in range(n_tokens - n + 1))
    return ngram_counter


def ngram_lengths_for(max_n: int) -> list[int]:
    """The multi-word n-gram orders to generate: ``[2, 3, ..., max_n]``.

    Unigrams are handled separately (they're just the tokens themselves), so
    n-gram generation starts at 2.
    """
    if max_n < MIN_NGRAM_LENGTH:
        return []
    return list(range(MIN_NGRAM_LENGTH, max_n + 1))


def clean_and_tokenize(document: str, max_n: int = DEFAULT_MAX_N) -> tuple[list[str], Counter[str]]:
    """Tokenize a document and generate its n-grams in one call.

    Returns ``(tokens, ngrams)`` where ``tokens`` is the unigram list and
    ``ngrams`` is a ``Counter`` of multi-word phrases (orders 2..max_n).
    """
    tokens = tokenize(document)
    ngrams = generate_ngrams(tokens, ngram_lengths_for(max_n))
    return tokens, ngrams


__all__ = [
    "DEFAULT_MAX_N",
    "MIN_NGRAM_LENGTH",
    "TOKEN_CLEAN_RE",
    "clean_and_tokenize",
    "generate_ngrams",
    "ngram_lengths_for",
    "tokenize",
]
