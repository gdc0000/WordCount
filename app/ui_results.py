import streamlit as st

from .plots import generate_barplot


def _infer_categories(enhanced_dataset):
    suffix = "_detected_words"
    categories = []
    for col in enhanced_dataset.columns:
        if col.endswith(suffix):
            categories.append(col[: -len(suffix)])
    return categories


def render_results(enhanced_dataset, selected_categories):
    st.subheader("📈 Enhanced Dataset Preview")
    try:
        st.dataframe(enhanced_dataset.head())
    except Exception as exc:
        st.error(f"Error displaying DataFrame: {exc}")

    csv = enhanced_dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Enhanced Dataset (CSV)",
        data=csv,
        file_name="enhanced_dataset.csv",
        mime="text/csv",
    )

    st.subheader("📊 Word Frequency Analysis")

    categories = selected_categories or _infer_categories(enhanced_dataset)
    if not categories:
        st.info("🔎 No categories available for plotting.")
        return

    plot_categories = st.multiselect(
        "🔎 Select Categories to Display Bar Plots:",
        options=categories,
        default=categories[:3] if len(categories) >= 3 else categories,
    )

    if plot_categories:
        for i in range(0, len(plot_categories), 3):
            cols = st.columns(3)
            for j, category in enumerate(plot_categories[i : i + 3]):
                with cols[j]:
                    st.markdown(f"**{category}**")
                    top_n = st.number_input(
                        f"Select number of top words/phrases to display for '{category}':",
                        min_value=1,
                        max_value=50,
                        value=3,
                        step=1,
                        key=f"top_n_{category}",
                    )
                    column_name_dw = f"{category}_detected_words"
                    if column_name_dw in enhanced_dataset.columns:
                        generate_barplot(
                            enhanced_dataset[column_name_dw], category, top_n=int(top_n)
                        )
                    else:
                        st.warning(f"No detected words data available for category '{category}'.")
    else:
        st.info("🔎 Please select at least one category to display bar plots.")
