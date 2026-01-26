import streamlit as st

from .stats_ui import perform_anova, perform_pearson_correlation


def render_stats_section(enhanced_dataset) -> None:
    st.markdown("---")
    st.subheader("📊 Statistical Analyses")

    analysis_tab, anova_tab = st.tabs(["🧵 Pearson Correlation", "📉 ANOVA"])
    with analysis_tab:
        perform_pearson_correlation(enhanced_dataset)
    with anova_tab:
        perform_anova(enhanced_dataset)
