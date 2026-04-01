import streamlit as st

from .plots import generate_barplot


def _normalize_export_column(column_name):
    if column_name is None:
        return None
    return "".join(ch for ch in str(column_name).replace(" ", "_") if ch.isalnum() or ch == "_")


def _infer_categories(enhanced_dataset):
    suffix = "_detected_words"
    categories = []
    for col in enhanced_dataset.columns:
        if col.endswith(suffix):
            categories.append(col[: -len(suffix)])
    return categories


def render_results(enhanced_dataset, selected_categories, selected_text_column=None):
    st.subheader("Enhanced Dataset Preview")
    try:
        st.dataframe(enhanced_dataset.head())
    except Exception as exc:
        st.error(f"Error displaying DataFrame: {exc}")

    export_dataset = enhanced_dataset
    export_text_column = _normalize_export_column(selected_text_column)

    if export_text_column in enhanced_dataset.columns:
        drop_text_column = st.checkbox(
            f"Exclude text column '{export_text_column}' from CSV export",
            value=False,
            help="Useful when the original text is very long and makes CSV or Excel files heavy.",
        )
        if drop_text_column:
            export_dataset = enhanced_dataset.drop(columns=[export_text_column])
            st.caption(f"CSV export will exclude '{export_text_column}'.")

    csv = export_dataset.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Enhanced Dataset (CSV)",
        data=csv,
        file_name="enhanced_dataset.csv",
        mime="text/csv",
    )

    st.subheader("Word Frequency Analysis")

    categories = selected_categories or _infer_categories(enhanced_dataset)
    if not categories:
        st.info("No categories available for plotting.")
        return

    plot_categories = st.multiselect(
        "Select Categories to Display Bar Plots:",
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
        st.info("Please select at least one category to display bar plots.")
