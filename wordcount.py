import streamlit as st

from app.data_io import load_dataset, load_wordlists
from app.enhance import enhance_dataset
from app.text_analysis import analyze_text_cached, normalize_wordlists_for_cache
from app.ui_controls import (
    render_category_selector,
    render_start_analysis_button,
    render_text_column_selector,
    render_wordlist_prefixes,
)
from app.ui_footer import add_footer
from app.ui_intro import render_intro
from app.ui_preview import render_dataset_preview, render_wordlists_summary
from app.ui_results import render_results
from app.ui_stats import render_stats_section
from app.ui_uploads import render_upload_section


st.set_page_config(page_title="WordCount Statistics", layout="wide")


def _ensure_session_state():
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "enhanced_dataset" not in st.session_state:
        st.session_state.enhanced_dataset = None
    if "selected_text_column" not in st.session_state:
        st.session_state.selected_text_column = None
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = None
    if "wordlists_bundle" not in st.session_state:
        st.session_state.wordlists_bundle = None
    if "wordlists_key" not in st.session_state:
        st.session_state.wordlists_key = None


def _wordlists_key(uploaded_wordlists, prefixes):
    return tuple(
        sorted((uploaded_file.name, prefixes.get(uploaded_file.name, "")) for uploaded_file in uploaded_wordlists)
    )


def main():
    _ensure_session_state()
    render_intro()

    uploaded_dataset, uploaded_wordlists = render_upload_section()

    if uploaded_dataset and uploaded_wordlists:
        dataset = load_dataset(uploaded_dataset)
        if dataset is None:
            add_footer()
            return

        prefixes, submitted = render_wordlist_prefixes(uploaded_wordlists)
        if submitted:
            st.session_state.wordlists_bundle = load_wordlists(uploaded_wordlists, prefixes)
            st.session_state.wordlists_key = _wordlists_key(uploaded_wordlists, prefixes)

        if st.session_state.wordlists_bundle:
            (
                exact_single_words,
                wildcard_single_prefixes,
                exact_multi_words,
                wildcard_multi_prefixes,
                summaries,
            ) = st.session_state.wordlists_bundle

            if exact_single_words:
                render_dataset_preview(dataset)
                render_wordlists_summary(summaries)

                selected_categories = render_category_selector(list(exact_single_words.keys()))
                st.session_state.selected_categories = selected_categories

                if selected_categories:
                    text_column = render_text_column_selector(dataset)
                    st.session_state.selected_text_column = text_column
                    if text_column and render_start_analysis_button():
                        with st.spinner(
                            "?? Performing Textual Analysis... This may take a while for large datasets."
                        ):
                            documents = dataset[text_column].astype(str).tolist()

                            selected_exact_single = {
                                cat: exact_single_words[cat] for cat in selected_categories
                            }
                            selected_wildcard_single = {
                                cat: wildcard_single_prefixes[cat] for cat in selected_categories
                            }
                            selected_exact_multi = {
                                cat: exact_multi_words[cat] for cat in selected_categories
                            }
                            selected_wildcard_multi = {
                                cat: wildcard_multi_prefixes[cat] for cat in selected_categories
                            }

                            normalized = normalize_wordlists_for_cache(
                                selected_exact_single,
                                selected_wildcard_single,
                                selected_exact_multi,
                                selected_wildcard_multi,
                            )
                            analysis_df = analyze_text_cached(documents, *normalized)
                            enhanced_dataset = enhance_dataset(dataset, analysis_df)
                            st.session_state.enhanced_dataset = enhanced_dataset
                            st.session_state.analysis_done = True
                            st.success("? Textual Analysis Completed!")
            else:
                st.error("No valid wordlists loaded. Please check your files.")
        else:
            st.info("Apply prefixes to load your wordlists.")

    if st.session_state.analysis_done and st.session_state.enhanced_dataset is not None:
        render_results(
            st.session_state.enhanced_dataset,
            st.session_state.selected_categories,
            st.session_state.selected_text_column,
        )
        render_stats_section(st.session_state.enhanced_dataset)

    add_footer()


if __name__ == "__main__":
    main()
