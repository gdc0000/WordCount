import re


def sanitize_identifier(value: str, fallback: str = "wordlist") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned or fallback


def uniquify(value: str, existing: set) -> str:
    if value not in existing:
        return value
    index = 2
    while f"{value}_{index}" in existing:
        index += 1
    return f"{value}_{index}"
