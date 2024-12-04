import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Tuple
from collections import Counter
import plotly.express as px
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io

# =========================
#        FUNCTIONS
# =========================

@st.cache_data
def load_dataset(uploaded_file):
    """
    Load a dataset from a CSV or Excel file.
    
    Parameters:
        uploaded_file: Uploaded file object from Streamlit.
        
    Returns:
        pandas.DataFrame or None
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format for dataset. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_wordlist(uploaded_file) -> Tuple[Dict[str, set], Dict[str, List[str]], Dict[str, set], Dict[str, List[str]]]:
    """
    Load and preprocess a multi-category wordlist from a CSV, TXT, DIC, DICX, or XLSX file.
    
    Parameters:
        uploaded_file: Uploaded file object from Streamlit.
        
    Returns:
        Tuple containing:
            - Dictionary mapping categories to exact single words
            - Dictionary mapping categories to wildcard prefixes for single words
            - Dictionary mapping categories to exact multi-word expressions
            - Dictionary mapping categories to wildcard prefixes for multi-word expressions
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            wordlist_df = pd.read_csv(uploaded_file)
        elif file_extension in ['txt', 'dic']:
            # Assume tab-separated for TXT and DIC files
            wordlist_df = pd.read_csv(uploaded_file, sep='\t')
        elif file_extension == 'dicx':
            # Treat .dicx as comma-separated
            wordlist_df = pd.read_csv(uploaded_file, sep=',')
        elif file_extension in ['xls', 'xlsx']:
            wordlist_df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format for wordlist. Please upload a CSV, TXT, DIC, DICX, or Excel file.")
            return None, None, None, None
        
        if 'DicTerm' not in wordlist_df.columns:
            st.error("The wordlist file must contain a column named 'DicTerm'.")
            return None, None, None, None
        
        # Identify category columns (excluding 'DicTerm')
        category_columns = [col for col in wordlist_df.columns if col != 'DicTerm']
        if not category_columns:
            st.error("No category columns found in the wordlist file.")
            return None, None, None, None
        
        # Initialize dictionaries
        exact_single_words = {category: set() for category in category_columns}
        wildcard_single_prefixes = {category: [] for category in category_columns}
        exact_multi_words = {category: set() for category in category_columns}
        wildcard_multi_prefixes = {category: [] for category in category_columns}
        
        # Iterate over each row and assign words to categories
        for _, row in wordlist_df.iterrows():
            term = str(row['DicTerm']).strip().lower()
            is_multi_word = len(term.split()) > 1  # Determine if the term is multi-word
            for category in category_columns:
                cell_value = row[category]
                if pd.notna(cell_value) and str(cell_value).strip().upper() == 'X':
                    if term.endswith('*'):
                        prefix = term[:-1].strip()
                        if prefix:  # Avoid empty prefix
                            if is_multi_word:
                                wildcard_multi_prefixes[category].append(prefix)
                            else:
                                wildcard_single_prefixes[category].append(prefix)
                    else:
                        if is_multi_word:
                            exact_multi_words[category].add(term)
                        else:
                            exact_single_words[category].add(term)
        
        return exact_single_words, wildcard_single_prefixes, exact_multi_words, wildcard_multi_prefixes
    except Exception as e:
        st.error(f"Error loading wordlist: {e}")
        return None, None, None, None

def clean_and_tokenize(document: str, max_n: int = 5) -> Tuple[List[str], List[str]]:
    """
    Clean and tokenize a document, generating both unigrams and n-grams up to max_n.
    
    Parameters:
        document: The text document as a string.
        max_n: Maximum number of words in n-grams to generate.
        
    Returns:
        Tuple containing:
            - List of unigrams (tokens).
            - List of n-grams (from bigrams up to max_n).
    """
    # Replace non-word characters with space and lowercase
    clean_doc = re.sub(r"[^\w\s']", ' ', document.lower())
    tokens = clean_doc.split()
    
    # Generate n-grams
    ngrams = []
    for n in range(2, max_n + 1):
        n_grams = zip(*[tokens[i:] for i in range(n)])
        ngrams += [' '.join(gram) for gram in n_grams]
    
    return tokens, ngrams

def count_words(tokens: List[str], ngrams: List[str],
               exact_single_words: set, wildcard_single_prefixes: List[str],
               exact_multi_words: set, wildcard_multi_prefixes: List[str]) -> Tuple[int, List[str]]:
    """
    Count words and n-grams in tokens and ngrams based on exact matches and wildcard prefixes.
    
    Parameters:
        tokens: List of tokens from the document.
        ngrams: List of n-grams from the document.
        exact_single_words: Set of exact single-word matches.
        wildcard_single_prefixes: List of wildcard prefixes for single words.
        exact_multi_words: Set of exact multi-word matches.
        wildcard_multi_prefixes: List of wildcard prefixes for multi-word expressions.
        
    Returns:
        Tuple containing:
            - Total count of wordlist identified.
            - List of detected words and n-grams.
    """
    detected_words = set()
    count = 0
    
    # Exact single word matches
    exact_matches = exact_single_words.intersection(tokens)
    detected_words.update(exact_matches)
    count += len(exact_matches)
    
    # Wildcard single word matches
    for prefix in wildcard_single_prefixes:
        matches = [token for token in tokens if token.startswith(prefix)]
        detected_words.update(matches)
        count += len(matches)
    
    # Exact multi-word matches
    exact_multi_matches = exact_multi_words.intersection(ngrams)
    detected_words.update(exact_multi_matches)
    count += len(exact_multi_matches)
    
    # Wildcard multi-word matches
    for prefix in wildcard_multi_prefixes:
        matches = [ngram for ngram in ngrams if ngram.startswith(prefix)]
        detected_words.update(matches)
        count += len(matches)
    
    detected_words_list = list(detected_words)
    return count, detected_words_list

def analyze_text(documents: pd.Series,
                exact_single_words: Dict[str, set],
                wildcard_single_prefixes: Dict[str, List[str]],
                exact_multi_words: Dict[str, set],
                wildcard_multi_prefixes: Dict[str, List[str]],
                progress_bar) -> pd.DataFrame:
    """
    Analyze text in documents using multiple wordlists, supporting both unigrams and n-grams.
    
    Parameters:
        documents: pandas Series containing text data.
        exact_single_words: Dictionary mapping categories to exact single words.
        wildcard_single_prefixes: Dictionary mapping categories to wildcard prefixes for single words.
        exact_multi_words: Dictionary mapping categories to exact multi-word expressions.
        wildcard_multi_prefixes: Dictionary mapping categories to wildcard prefixes for multi-word expressions.
        progress_bar: Streamlit progress bar object.
        
    Returns:
        pandas.DataFrame with analysis results for each category and global metrics.
    """
    # Initialize lists for global metrics
    n_tokens_list = []
    n_types_list = []
    
    # Initialize a dictionary to hold results for each category
    analysis_results = {category: {
        'word_count': [],
        'word_perc': [],
        'detected_words': []
    } for category in exact_single_words.keys()}
    
    total_docs = len(documents)
    
    for i, doc in enumerate(documents):
        if pd.isna(doc):
            doc = ""
        tokens, ngrams = clean_and_tokenize(doc)
        n_tokens = len(tokens)
        n_types = len(set(tokens))
        
        # Append global metrics
        n_tokens_list.append(n_tokens)
        n_types_list.append(n_types)
        
        for category in exact_single_words.keys():
            count, detected = count_words(
                tokens,
                ngrams,
                exact_single_words[category],
                wildcard_single_prefixes[category],
                exact_multi_words[category],
                wildcard_multi_prefixes[category]
            )
            word_perc = count / n_tokens if n_tokens > 0 else 0.0
            analysis_results[category]['word_count'].append(count)
            analysis_results[category]['word_perc'].append(word_perc)
            analysis_results[category]['detected_words'].append(detected)
        
        # Update progress every 100 documents or at the end
        if (i + 1) % 100 == 0 or i == total_docs -1:
            progress = (i + 1) / total_docs
            progress_bar.progress(progress)
    
    # Create a DataFrame for global metrics
    global_metrics = pd.DataFrame({
        'n_tokens': n_tokens_list,
        'n_types': n_types_list
    })
    
    # Create DataFrames for each category
    category_metrics = {}
    for category, metrics in analysis_results.items():
        category_metrics[f"{category}_word_count"] = metrics['word_count']
        category_metrics[f"{category}_word_perc"] = metrics['word_perc']
        category_metrics[f"{category}_detected_words"] = metrics['detected_words']
    
    category_metrics_df = pd.DataFrame(category_metrics)
    
    # Combine global metrics with category metrics
    analysis_df = pd.concat([global_metrics, category_metrics_df], axis=1)
    return analysis_df

def enhance_dataset(dataset: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance the original dataset with analysis results.
    
    Parameters:
        dataset: Original pandas DataFrame.
        analysis_df: DataFrame containing analysis results.
        
    Returns:
        Enhanced pandas DataFrame with converted 'detected_words' columns.
    """
    dataset = dataset.reset_index(drop=True)
    enhanced_dataset = pd.concat([dataset, analysis_df], axis=1)

    # Identify columns that contain 'detected_words' and convert lists to strings
    detected_words_columns = [col for col in enhanced_dataset.columns if 'detected_words' in col]

    for col in detected_words_columns:
        enhanced_dataset[col] = enhanced_dataset[col].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else ''
        )

    return enhanced_dataset

def generate_summary_list(exact_single_words: Dict[str, set],
                          exact_multi_words: Dict[str, set]) -> str:
    """
    Generate a simple list summary of the wordlist categories, showing top three and bottom three words/phrases.
    
    Parameters:
        exact_single_words: Dictionary mapping categories to exact single words.
        exact_multi_words: Dictionary mapping categories to exact multi-word expressions.
        
    Returns:
        A formatted string representing the wordlist summary.
    """
    summary_lines = []
    for category in exact_single_words.keys():
        single_words = exact_single_words[category]
        multi_words = exact_multi_words[category]
        total_words = len(single_words) + len(multi_words)
        
        if total_words == 0:
            words_str = "None"
        else:
            # Combine single and multi-word terms
            combined_words = sorted(single_words) + sorted(multi_words)
            if total_words <= 6:
                words_str = ', '.join(combined_words)
            else:
                top_three = combined_words[:3]
                bottom_three = combined_words[-3:]
                words_str = ', '.join(top_three) + ', ... , ' + ', '.join(bottom_three)
        summary_line = f"**{category} ({total_words} terms):** {words_str};"
        summary_lines.append(summary_line)
    summary_text = '\n\n'.join(summary_lines)  # Double newline for better spacing
    return summary_text

def generate_barplot(detected_words_series: pd.Series, label: str, top_n: int = 3) -> None:
    """
    Generate and display a horizontal Plotly bar plot from detected words.
    
    Parameters:
        detected_words_series: pandas Series containing lists of detected words.
        label: Label/category name for the wordlist.
        top_n: Number of top words to display.
    """
    # Flatten the list of detected words
    all_detected_words = [word.strip() for sublist in detected_words_series for word in sublist.split(',') if word.strip()]
    if not all_detected_words:
        st.warning(f"No words detected for category '{label}' to generate a bar plot.")
        return
    
    # Count word frequencies
    word_counts = Counter(all_detected_words)
    
    # Get the top N words
    top_words = word_counts.most_common(top_n)
    words, counts = zip(*top_words)
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'Word': words,
        'Frequency': counts
    })
    
    # Generate horizontal bar plot using Plotly
    fig = px.bar(
        df_plot,
        x='Frequency',
        y='Word',
        orientation='h',
        title=f"📊 Top {top_n} Words/Phrases in '{label}' Category",
        labels={'Frequency': 'Frequency', 'Word': 'Word/Phrase'},
        height=400  # Reduced height for smaller plots
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    
    # Display the Plotly bar plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def perform_pearson_correlation(enhanced_df: pd.DataFrame):
    """
    Allow the user to compute Pearson correlation between two numeric columns.
    
    Parameters:
        enhanced_df: Enhanced pandas DataFrame with analysis results.
    """
    st.header("🔍 Pearson Correlation Analysis")
    
    # Select numeric columns for correlation
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
        # Drop NaN values
        df_clean = enhanced_df[[col1, col2]].dropna()
        if df_clean.empty:
            st.error("No data available to compute correlation.")
            return
        # Compute Pearson correlation
        corr_coef, p_value = pearsonr(df_clean[col1], df_clean[col2])
        st.write(f"**Pearson Correlation Coefficient:** {corr_coef:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        # Generate Plotly scatter plot with trendline
        fig = px.scatter(
            df_clean, 
            x=col1, 
            y=col2, 
            trendline="ols", 
            title=f"Scatter Plot of {col1} vs {col2} with Trendline",
            labels={col1: col1, col2: col2}
        )
        st.plotly_chart(fig, use_container_width=True)

def perform_anova(enhanced_df: pd.DataFrame):
    """
    Allow the user to perform ANOVA to test mean differences across groups.
    
    Parameters:
        enhanced_df: Enhanced pandas DataFrame with analysis results.
    """
    st.header("📊 ANOVA (Analysis of Variance)")
    
    # Identify categorical and numeric columns
    categorical_cols = enhanced_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = enhanced_df.select_dtypes(include=['number']).columns.tolist()
    
    if not categorical_cols:
        st.warning("No categorical columns available for ANOVA.")
        return
    if not numeric_cols:
        st.warning("No numeric columns available for ANOVA.")
        return
    
    # User selects categorical and numeric variables
    cat_var = st.selectbox("Select the categorical (between factor) variable:", options=categorical_cols, key="anova_cat_var")
    num_var = st.selectbox("Select the numeric dependent variable:", options=numeric_cols, key="anova_num_var")
    
    if st.button("Perform ANOVA"):
        # Drop NaN values
        df_clean = enhanced_df[[cat_var, num_var]].dropna()
        if df_clean.empty:
            st.error("No data available to perform ANOVA.")
            return
        
        # Ensure categorical variable is treated as category
        df_clean[cat_var] = df_clean[cat_var].astype('category')
        
        # Perform one-way ANOVA using statsmodels
        model = ols(f'{num_var} ~ C({cat_var})', data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        st.write("**ANOVA Table:**")
        st.table(anova_table)
        
        # Check if the ANOVA is significant
        if anova_table['PR(>F)'][0] < 0.05:
            st.success("The ANOVA test is significant (p < 0.05). There are significant differences between group means.")
        else:
            st.info("The ANOVA test is not significant (p ≥ 0.05). There are no significant differences between group means.")
        
        # Perform Tukey's HSD Post Hoc Test
        tukey = pairwise_tukeyhsd(endog=df_clean[num_var], groups=df_clean[cat_var], alpha=0.05)
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
