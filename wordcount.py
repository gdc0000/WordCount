import streamlit as st

from app.data_io import load_dataset, load_wordlist
from app.enhance import enhance_dataset
from app.text_analysis import analyze_text
from app.ui_controls import (
    render_category_selector,
    render_start_analysis_button,
    render_text_column_selector,
)
from app.ui_footer import add_footer
from app.ui_intro import render_intro
from app.ui_preview import render_dataset_preview, render_wordlist_summary
from app.ui_results import render_results
from app.ui_stats import render_stats_section
from app.ui_uploads import render_upload_section


st.set_page_config(page_title="WordCount Statistics", layout="wide")


def _ensure_session_state():
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "enhanced_dataset" not in st.session_state:
        st.session_state.enhanced_dataset = None
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = None


def main():
    _ensure_session_state()
    render_intro()

    uploaded_dataset, uploaded_wordlist = render_upload_section()

    if uploaded_dataset and uploaded_wordlist:
        dataset = load_dataset(uploaded_dataset)
        wordlist = load_wordlist(uploaded_wordlist)

        if dataset is not None and all(item is not None for item in wordlist):
            (
                exact_single_words,
                wildcard_single_prefixes,
                exact_multi_words,
                wildcard_multi_prefixes,
            ) = wordlist

            render_dataset_preview(dataset)
            render_wordlist_summary(exact_single_words, exact_multi_words)

            selected_categories = render_category_selector(list(exact_single_words.keys()))
            st.session_state.selected_categories = selected_categories

            if selected_categories:
                text_column = render_text_column_selector(dataset)
                if text_column and render_start_analysis_button():
                    with st.spinner(
                        "?? Performing Textual Analysis... This may take a while for large datasets."
                    ):
                        documents = dataset[text_column].astype(str)
                        progress_bar = st.progress(0)

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

                        analysis_df = analyze_text(
                            documents,
                            selected_exact_single,
                            selected_wildcard_single,
                            selected_exact_multi,
                            selected_wildcard_multi,
                            progress_bar,
                        )
                        enhanced_dataset = enhance_dataset(dataset, analysis_df)
                        st.session_state.enhanced_dataset = enhanced_dataset
                        st.session_state.analysis_done = True
                        st.success("? Textual Analysis Completed!")

    if st.session_state.analysis_done and st.session_state.enhanced_dataset is not None:
        render_results(st.session_state.enhanced_dataset, st.session_state.selected_categories)
        render_stats_section(st.session_state.enhanced_dataset)

    add_footer()


if __name__ == "__main__":
    main()
