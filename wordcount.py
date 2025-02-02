import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import plotly.express as px
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path

# =========================
#        PAGE CONFIG
# =========================
st.set_page_config(page_title="WordCount Statistics", layout="wide")

# =========================
#        HELPER FUNCTIONS
# =========================

def read_file(uploaded_file, file_mapping: Dict[str, dict]) -> Optional[pd.DataFrame]:
    """
    Read a file using the proper pandas reader based on file extension.
    """
    ext = Path(uploaded_file.name).suffix.lower()
    reader = file_mapping.get(ext)
    if reader:
        try:
            return reader(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file ({ext}): {e}")
            return None
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# =========================
#        DATA LOADING
# =========================

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load a dataset from a CSV, TSV, or Excel file.
    """
    file_mapping = {
        '.csv': lambda f: pd.read_csv(f),
        '.tsv': lambda f: pd.read_csv(f, sep='\t'),
        '.xls': lambda f: pd.read_excel(f),
        '.xlsx': lambda f: pd.read_excel(f),
    }
    return read_file(uploaded_file, file_mapping)

@st.cache_data(show_spinner=False)
def load_wordlist(uploaded_file) -> Tuple[Optional[Dict[str, set]], Optional[Dict[str, List[str]]],
                                           Optional[Dict[str, set]], Optional[Dict[str, List[str]]]]:
    """
    Load and preprocess a multi-category wordlist.
    Expected column: 'DicTerm'. Other columns represent categories (an 'X' marks membership).
    Supports CSV, TXT, DIC, DICX, and Excel files.
    """
    ext = Path(uploaded_file.name).suffix.lower()
    file_mapping = {
        '.csv': lambda f: pd.read_csv(f),
        '.txt': lambda f: pd.read_csv(f, sep='\t'),
        '.dic': lambda f: pd.read_csv(f, sep='\t'),
        '.dicx': lambda f: pd.read_csv(f, sep=','),
        '.xls': lambda f: pd.read_excel(f),
        '.xlsx': lambda f: pd.read_excel(f)
    }
    df = read_file(uploaded_file, file_mapping)
    if df is None:
        return None, None, None, None

    if 'DicTerm' not in df.columns:
        st.error("The wordlist file must contain a column named 'DicTerm'.")
        return None, None, None, None

    # Identify category columns and sanitize names
    category_columns = [col for col in df.columns if col != 'DicTerm']
    if not category_columns:
        st.error("No category columns found in the wordlist file.")
        return None, None, None, None

    sanitized_categories = [col.replace(' ', '_') for col in category_columns]
    df.columns = ['DicTerm'] + sanitized_categories

    # Initialize dictionaries for exact and wildcard terms
    exact_single_words = {cat: set() for cat in sanitized_categories}
    wildcard_single_prefixes = {cat: [] for cat in sanitized_categories}
    exact_multi_words = {cat: set() for cat in sanitized_categories}
    wildcard_multi_prefixes = {cat: [] for cat in sanitized_categories}

    for _, row in df.iterrows():
        term = str(row['DicTerm']).strip().lower()
        is_multi_word = len(term.split()) > 1
        for cat in sanitized_categories:
            cell_value = row[cat]
            if pd.notna(cell_value) and str(cell_value).strip().upper() == 'X':
                if term.endswith('*'):
                    prefix = term[:-1].strip()
                    if prefix:
                        if is_multi_word:
                            wildcard_multi_prefixes[cat].append(prefix)
                        else:
                            wildcard_single_prefixes[cat].append(prefix)
                else:
                    if is_multi_word:
                        exact_multi_words[cat].add(term)
                    else:
                        exact_single_words[cat].add(term)
    return exact_single_words, wildcard_single_prefixes, exact_multi_words, wildcard_multi_prefixes

# =========================
#        TEXT PROCESSING
# =========================

# Precompile regex for performance
TOKEN_REGEX = re.compile(r"[^\w\s']")

def clean_and_tokenize(document: str, max_n: int = 5) -> Tuple[List[str], List[str]]:
    """
    Clean and tokenize a document into unigrams and n-grams.
    """
    clean_doc = TOKEN_REGEX.sub(' ', document.lower())
    tokens = clean_doc.split()
    
    # Generate n-grams (bigrams to max_n-grams)
    ngrams = [' '.join(tokens[i:i+n]) for n in range(2, max_n + 1) for i in range(len(tokens) - n + 1)]
    return tokens, ngrams

def count_words(tokens: List[str], ngrams: List[str],
                exact_single: set, wildcard_single: List[str],
                exact_multi: set, wildcard_multi: List[str]) -> Tuple[int, List[str]]:
    """
    Count matching words/phrases in tokens and n-grams.
    """
    detected = set()
    count = 0

    # Exact single word matches
    matches = exact_single.intersection(tokens)
    detected.update(matches)
    count += len(matches)

    # Wildcard single matches
    for prefix in wildcard_single:
        found = [token for token in tokens if token.startswith(prefix)]
        detected.update(found)
        count += len(found)

    # Exact multi-word matches
    matches = exact_multi.intersection(ngrams)
    detected.update(matches)
    count += len(matches)

    # Wildcard multi-word matches
    for prefix in wildcard_multi:
        found = [ng for ng in ngrams if ng.startswith(prefix)]
        detected.update(found)
        count += len(found)

    return count, list(detected)

def analyze_text(documents: pd.Series,
                 exact_single_words: Dict[str, set],
                 wildcard_single_prefixes: Dict[str, List[str]],
                 exact_multi_words: Dict[str, set],
                 wildcard_multi_prefixes: Dict[str, List[str]],
                 progress_bar) -> pd.DataFrame:
    """
    Analyze each document for wordlist matches.
    """
    n_tokens_list, n_types_list = [], []
    # Initialize results per category
    results = {cat: {'word_count': [], 'word_perc': [], 'detected_words': []}
               for cat in exact_single_words.keys()}
    
    total_docs = len(documents)
    for i, doc in enumerate(documents):
        doc = str(doc) if pd.notna(doc) else ""
        tokens, ngrams = clean_and_tokenize(doc)
        n_tokens = len(tokens)
        n_types = len(set(tokens))
        n_tokens_list.append(n_tokens)
        n_types_list.append(n_types)

        for cat in exact_single_words.keys():
            cnt, detected = count_words(
                tokens,
                ngrams,
                exact_single_words[cat],
                wildcard_single_prefixes[cat],
                exact_multi_words[cat],
                wildcard_multi_prefixes[cat]
            )
            word_perc = cnt / n_tokens if n_tokens > 0 else 0.0
            results[cat]['word_count'].append(cnt)
            results[cat]['word_perc'].append(word_perc)
            results[cat]['detected_words'].append(detected)

        # Update progress bar (update more frequently if dataset is small)
        if total_docs < 100 or (i + 1) % 100 == 0 or i == total_docs - 1:
            progress_bar.progress((i + 1) / total_docs)

    global_df = pd.DataFrame({'n_tokens': n_tokens_list, 'n_types': n_types_list})
    cat_data = {}
    for cat, metrics in results.items():
        cat_data[f"{cat}_word_count"] = metrics['word_count']
        cat_data[f"{cat}_word_perc"] = metrics['word_perc']
        cat_data[f"{cat}_detected_words"] = metrics['detected_words']
    cat_df = pd.DataFrame(cat_data)
    return pd.concat([global_df, cat_df], axis=1)

def enhance_dataset(dataset: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the original dataset with analysis results and clean column names.
    """
    dataset = dataset.reset_index(drop=True)
    enhanced = pd.concat([dataset, analysis_df], axis=1)

    # Convert any list/dict/set columns to comma-separated strings for display/export
    def convert(x):
        if isinstance(x, (list, set)):
            return ', '.join(map(str, x))
        elif isinstance(x, dict):
            return ', '.join([f"{k}: {v}" for k, v in x.items()])
        return x

    for col in enhanced.columns:
        if enhanced[col].apply(lambda x: isinstance(x, (list, dict, set))).any():
            enhanced[col] = enhanced[col].apply(convert)

    # Sanitize column names: replace spaces and special characters with underscores
    enhanced.columns = enhanced.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    return enhanced

def generate_summary_list(exact_single_words: Dict[str, set],
                          exact_multi_words: Dict[str, set]) -> str:
    """
    Generate a summary string listing the top three and bottom three words/phrases per category.
    """
    summary_lines = []
    for cat in exact_single_words.keys():
        single = exact_single_words[cat]
        multi = exact_multi_words[cat]
        total = len(single) + len(multi)
        if total == 0:
            words_str = "None"
        else:
            combined = sorted(single) + sorted(multi)
            if total <= 6:
                words_str = ', '.join(combined)
            else:
                words_str = ', '.join(combined[:3]) + ', ... , ' + ', '.join(combined[-3:])
        summary_lines.append(f"**{cat} ({total} terms):** {words_str};")
    return "\n\n".join(summary_lines)

def generate_barplot(detected_words_series: pd.Series, label: str, top_n: int = 3) -> None:
    """
    Generate a horizontal bar plot for the top detected words/phrases in a given category.
    """
    # Convert detected words (stored as comma-separated strings) back to list
    words_list = []
    for entry in detected_words_series.dropna():
        # If the entry is already a list, use it; otherwise, split by comma
        if isinstance(entry, list):
            words_list.extend(entry)
        elif isinstance(entry, str):
            words_list.extend([word.strip() for word in entry.split(',') if word.strip()])
    if not words_list:
        st.warning(f"No words detected for category '{label}' to generate a bar plot.")
        return

    counts = Counter(words_list)
    top = counts.most_common(top_n)
    if not top:
        st.warning(f"No words detected for category '{label}' to generate a bar plot.")
        return
    df_plot = pd.DataFrame(top, columns=['Word', 'Frequency'])
    fig = px.bar(
        df_plot,
        x='Frequency',
        y='Word',
        orientation='h',
        title=f"📊 Top {top_n} Words/Phrases in '{label}' Category",
        height=400
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# =========================
#        STATISTICAL ANALYSES
# =========================

def perform_pearson_correlation(enhanced_df: pd.DataFrame):
    st.header("🔍 Pearson Correlation Analysis")
    numeric_cols = enhanced_df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns available for correlation analysis.")
        return

    col1 = st.selectbox("Select the first numeric column:", options=numeric_cols, key="corr_col1")
    col2 = st.selectbox("Select the second numeric column:", options=numeric_cols, index=1, key="corr_col2")
    if st.button("Compute Pearson Correlation"):
        if col1 == col2:
            st.error("Please select two different columns.")
            return
        df_clean = enhanced_df[[col1, col2]].dropna().copy()
        if df_clean.empty:
            st.error("No data available to compute correlation.")
            return
        coef, p_val = pearsonr(df_clean[col1], df_clean[col2])
        st.write(f"**Pearson Correlation Coefficient:** {coef:.4f}")
        st.write(f"**P-value:** {p_val:.4f}")

        fig = px.scatter(df_clean, x=col1, y=col2, trendline="ols",
                         title=f"Scatter Plot of {col1} vs {col2} with Trendline",
                         labels={col1: col1, col2: col2})
        st.plotly_chart(fig, use_container_width=True)

def perform_anova(enhanced_df: pd.DataFrame):
    st.header("📊 ANOVA (Analysis of Variance)")
    categorical_cols = enhanced_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = enhanced_df.select_dtypes(include=['number']).columns.tolist()

    if not categorical_cols:
        st.warning("No categorical columns available for ANOVA.")
        return
    if not numeric_cols:
        st.warning("No numeric columns available for ANOVA.")
        return

    cat_var = st.selectbox("Select the categorical variable:", options=categorical_cols, key="anova_cat_var")
    num_var = st.selectbox("Select the numeric dependent variable:", options=numeric_cols, key="anova_num_var")
    if st.button("Perform ANOVA"):
        df_clean = enhanced_df[[cat_var, num_var]].dropna().copy()
        if df_clean.empty:
            st.error("No data available to perform ANOVA.")
            return

        df_clean[cat_var] = df_clean[cat_var].astype('category')
        try:
            model = ols(f'{num_var} ~ C({cat_var})', data=df_clean).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.write("**ANOVA Table:**")
            st.table(anova_table)

            p_val = anova_table['PR(>F)'][0]
            if p_val < 0.05:
                st.success("Significant differences detected between groups (p < 0.05).")
            else:
                st.info("No significant differences detected between groups (p ≥ 0.05).")

            tukey = pairwise_tukeyhsd(endog=df_clean[num_var], groups=df_clean[cat_var], alpha=0.05)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            st.write("**Tukey's HSD Post Hoc Comparisons:**")
            st.table(tukey_df)

            # Using error bars in Plotly for confidence intervals
            group_stats = df_clean.groupby(cat_var)[num_var].agg(['mean', 'sem']).reset_index()
            group_stats.rename(columns={'sem': 'se'}, inplace=True)
            fig = px.bar(
                group_stats,
                x='mean',
                y=cat_var,
                orientation='h',
                error_x='se',
                title=f'Bar Plot of {num_var} by {cat_var} with 95% Confidence Intervals',
                labels={'mean': f'Mean of {num_var}', cat_var: cat_var},
                height=600
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error performing ANOVA: {e}")

# =========================
#        FOOTER
# =========================

def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

# =========================
#        STREAMLIT APP
# =========================

def main():
    # Session state initialization and reset option
    if st.sidebar.button("🔄 Reset Analysis"):
        st.session_state.clear()
        st.experimental_rerun()

    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'enhanced_dataset' not in st.session_state:
        st.session_state.enhanced_dataset = None

    st.title("📊 WordCount Statistics")
    st.markdown("""
    ## 👋 Welcome to WordCount Statistics!
    
    This tool allows you to analyze textual data using customizable wordlists. Follow the instructions below:
    
    1. **Upload Files** via the sidebar.
    2. **Preview** your dataset and review the wordlist summary.
    3. **Select Categories** and the text column to analyze.
    4. **Perform Analysis** and view the enhanced dataset, visualizations, and statistical outputs.
    """)

    # Sidebar file uploads
    st.sidebar.header("📥 Upload Files")
    uploaded_dataset = st.sidebar.file_uploader("Upload your dataset (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])
    uploaded_wordlist = st.sidebar.file_uploader("Upload your wordlist (CSV, TXT, DIC, DICX, XLS, XLSX)", type=["csv", "txt", "dic", "dicx", "xls", "xlsx"])

    if uploaded_dataset and uploaded_wordlist:
        dataset = load_dataset(uploaded_dataset)
        ws = load_wordlist(uploaded_wordlist)
        if dataset is not None and all(ws):
            exact_single_words, wildcard_single_prefixes, exact_multi_words, wildcard_multi_prefixes = ws

            st.subheader("📄 Dataset Preview")
            st.dataframe(dataset.head())

            st.subheader("📃 Wordlist Summary")
            summary_text = generate_summary_list(exact_single_words, exact_multi_words)
            st.markdown(summary_text)

            # Category selection
            selected_categories = st.multiselect("📌 Select Categories to Analyze",
                                                 options=list(exact_single_words.keys()),
                                                 default=list(exact_single_words.keys()))
            if selected_categories:
                text_columns = dataset.select_dtypes(include=['object', 'string']).columns.tolist()
                if not text_columns:
                    st.error("No text columns found in the dataset.")
                else:
                    text_column = st.selectbox("🔍 Select the text column:", text_columns)
                    if st.button("🚀 Start Analysis"):
                        with st.spinner("Performing textual analysis..."):
                            docs = dataset[text_column].astype(str)
                            progress_bar = st.progress(0)
                            # Filter wordlists based on selected categories
                            sel_exact_single = {cat: exact_single_words[cat] for cat in selected_categories}
                            sel_wildcard_single = {cat: wildcard_single_prefixes[cat] for cat in selected_categories}
                            sel_exact_multi = {cat: exact_multi_words[cat] for cat in selected_categories}
                            sel_wildcard_multi = {cat: wildcard_multi_prefixes[cat] for cat in selected_categories}

                            analysis_df = analyze_text(
                                docs,
                                sel_exact_single,
                                sel_wildcard_single,
                                sel_exact_multi,
                                sel_wildcard_multi,
                                progress_bar
                            )
                            enhanced_dataset = enhance_dataset(dataset, analysis_df)
                            st.session_state.enhanced_dataset = enhanced_dataset
                            st.session_state.analysis_done = True
                            st.success("✅ Textual Analysis Completed!")

    # Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.enhanced_dataset is not None:
        enhanced_dataset = st.session_state.enhanced_dataset

        st.subheader("📈 Enhanced Dataset Preview")
        try:
            st.dataframe(enhanced_dataset.head())
        except Exception as e:
            st.error(f"Error displaying DataFrame: {e}")

        csv = enhanced_dataset.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Enhanced Dataset (CSV)",
            data=csv,
            file_name="enhanced_dataset.csv",
            mime="text/csv",
        )

        st.subheader("📊 Word Frequency Analysis")
        plot_categories = st.multiselect("🔎 Select Categories for Bar Plots:",
                                         options=selected_categories,
                                         default=selected_categories[:3] if len(selected_categories) >= 3 else selected_categories)
        if plot_categories:
            for i in range(0, len(plot_categories), 3):
                cols = st.columns(3)
                for j, cat in enumerate(plot_categories[i:i+3]):
                    with cols[j]:
                        st.markdown(f"**{cat}**")
                        top_n = st.number_input(f"Top words/phrases for '{cat}':", min_value=1, max_value=50, value=3, step=1, key=f"top_n_{cat}")
                        col_detect = f"{cat}_detected_words"
                        if col_detect in enhanced_dataset.columns:
                            generate_barplot(enhanced_dataset[col_detect], cat, top_n=int(top_n))
                        else:
                            st.warning(f"No detected words for '{cat}'.")
        else:
            st.info("Please select at least one category for bar plots.")

        st.markdown("---")
        st.subheader("📊 Statistical Analyses")
        tab_corr, tab_anova = st.tabs(["🔗 Pearson Correlation", "📉 ANOVA"])
        with tab_corr:
            perform_pearson_correlation(enhanced_dataset)
        with tab_anova:
            perform_anova(enhanced_dataset)

    add_footer()

if __name__ == "__main__":
    main()
