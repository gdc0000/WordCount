from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


def generate_barplot(detected_words_series: pd.Series, label: str, top_n: int = 3) -> None:
    """
    Generate and display a horizontal Plotly bar plot from detected words.
    """
    all_detected_words = [
        word.strip()
        for sublist in detected_words_series.dropna()
        for word in sublist.split(",")
        if word.strip()
    ]
    if not all_detected_words:
        st.warning(f"No words detected for category '{label}' to generate a bar plot.")
        return

    word_counts = Counter(all_detected_words)
    top_words = word_counts.most_common(top_n)
    if not top_words:
        st.warning(f"No words detected for category '{label}' to generate a bar plot.")
        return
    words, counts = zip(*top_words)

    df_plot = pd.DataFrame({"Word": words, "Frequency": counts})

    fig = px.bar(
        df_plot,
        x="Frequency",
        y="Word",
        orientation="h",
        title=f"📊 Top {top_n} Words/Phrases in '{label}' Category",
        labels={"Frequency": "Frequency", "Word": "Word/Phrase"},
        height=400,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
