from typing import Dict


def generate_summary_list(exact_single_words: Dict[str, set], exact_multi_words: Dict[str, set]) -> str:
    """
    Generate a list summary of the wordlist categories.
    """
    summary_lines = []
    for category in exact_single_words.keys():
        single_words = exact_single_words[category]
        multi_words = exact_multi_words[category]
        total_words = len(single_words) + len(multi_words)

        if total_words == 0:
            words_str = "None"
        else:
            combined_words = sorted(single_words) + sorted(multi_words)
            if total_words <= 6:
                words_str = ", ".join(combined_words)
            else:
                top_three = combined_words[:3]
                bottom_three = combined_words[-3:]
                words_str = ", ".join(top_three) + ", ... , " + ", ".join(bottom_three)
        summary_lines.append(f"**{category} ({total_words} terms):** {words_str};")
    return "\n\n".join(summary_lines)
