import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Tuple
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
def load_wordlist(uploaded_file) -> Tuple[Dict[str, set], Dict[str, List[str]]]:
    """
    Load and preprocess a multi-category wordlist from a CSV, TXT, DIC, DICX, or XLSX file.
    
    Parameters:
        uploaded_file: Uploaded file object from Streamlit.
        
    Returns:
        Tuple containing:
            - Dictionary mapping categories to exact words
            - Dictionary mapping categories to wildcard prefixes
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
            return None, None
        
        if 'DicTerm' not in wordlist_df.columns:
            st.error("The wordlist file must contain a column named 'DicTerm'.")
            return None, None
        
        # Identify category columns (excluding 'DicTerm')
        category_columns = [col for col in wordlist_df.columns if col != 'DicTerm']
        if not category_columns:
            st.error("No category columns found in the wordlist file.")
            return None, None
        
        # Initialize dictionaries
        exact_words = {category: set() for category in category_columns}
        wildcard_prefixes = {category: [] for category in category_columns}
        
        # Iterate over each row and assign words to categories
        for _, row in wordlist_df.iterrows():
            word = str(row['DicTerm']).strip().lower()
            for category in category_columns:
                cell_value = row[category]
                if pd.notna(cell_value) and str(cell_value).strip().upper() == 'X':
                    if word.endswith('*'):
                        prefix = word[:-1]
                        if prefix:  # Avoid empty prefix
                            wildcard_prefixes[category].append(prefix)
                    else:
                        exact_words[category].add(word)
        
        return exact_words, wildcard_prefixes
    except Exception as e:
        st.error(f"Error loading wordlist: {e}")
        return None, None

def clean_and_tokenize(document: str) -> List[str]:
    """
    Clean and tokenize a document.
    
    Parameters:
        document: The text document as a string.
        
    Returns:
        List of tokens.
    """
    # Replace non-word characters with space and lowercase
    clean_doc = re.sub(r"[^\w\s']", ' ', document.lower())
    tokens = clean_doc.split()
    return tokens

def count_words(tokens: List[str], exact_words: set, wildcard_prefixes: List[str]) -> Tuple[int, List[str]]:
    """
    Count words in tokens based on exact matches and wildcard prefixes.
    
    Parameters:
        tokens: List of tokens from the document.
        exact_words: Set of exact words to match.
        wildcard_prefixes: List of prefixes for wildcard matching.
        
    Returns:
        Tuple containing:
            - Count of wordlist identified
            - List of detected words
    """
    detected_words = set()
    count = 0
    
    # Exact word matches
    exact_matches = exact_words.intersection(tokens)
    detected_words.update(exact_matches)
    count += len(exact_matches)
    
    # Wildcard prefix matches
    for prefix in wildcard_prefixes:
        matches = [token for token in tokens if token.startswith(prefix)]
        detected_words.update(matches)
        count += len(matches)
    
    detected_words_list = list(detected_words)
    return count, detected_words_list

def analyze_text(documents: pd.Series, exact_words: Dict[str, set], wildcard_prefixes: Dict[str, List[str]], progress_bar) -> pd.DataFrame:
    """
    Analyze text in documents using multiple wordlists.
    
    Parameters:
        documents: pandas Series containing text data.
        exact_words: Dictionary mapping categories to exact words.
        wildcard_prefixes: Dictionary mapping categories to wildcard prefixes.
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
    } for category in exact_words.keys()}
    
    total_docs = len(documents)
    
    for i, doc in enumerate(documents):
        if pd.isna(doc):
            doc = ""
        tokens = clean_and_tokenize(doc)
        n_tokens = len(tokens)
        n_types = len(set(tokens))
        
        # Append global metrics
        n_tokens_list.append(n_tokens)
        n_types_list.append(n_types)
        
        for category in exact_words.keys():
            ew = exact_words[category]
            wp = wildcard_prefixes[category]
            count, detected = count_words(tokens, ew, wp)
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
        Enhanced pandas DataFrame.
    """
    dataset = dataset.reset_index(drop=True)
    enhanced_dataset = pd.concat([dataset, analysis_df], axis=1)
    return enhanced_dataset

def generate_wordcloud(detected_words_series: pd.Series, label: str) -> None:
    """
    Generate and display a word cloud from detected words.
    
    Parameters:
        detected_words_series: pandas Series containing lists of detected words.
        label: Label/category name for the wordlist.
    """
    # Flatten the list of detected words
    all_detected_words = [word for sublist in detected_words_series for word in sublist]
    if not all_detected_words:
        st.warning(f"No words detected for category '{label}' to generate a word cloud.")
        return
    
    # Count word frequencies
    word_counts = Counter(all_detected_words)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
    
    # Plotting the word cloud using matplotlib
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"🖼️ Word Cloud for '{label}'", fontsize=20)
    plt.tight_layout(pad=0)
    
    # Display the word cloud in Streamlit
    st.pyplot(plt)

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
    
    col1, col2 = st.selectbox("Select the first numeric column:", options=numeric_cols), st.selectbox("Select the second numeric column:", options=numeric_cols, index=1)
    
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
        fig = px.scatter(df_clean, x=col1, y=col2, trendline="ols", title=f"Scatter Plot of {col1} vs {col2} with Trendline")
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
    cat_var = st.selectbox("Select the categorical (between factor) variable:", options=categorical_cols)
    num_var = st.selectbox("Select the numeric dependent variable:", options=numeric_cols)
    
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
        
        # Generate Plotly bar plot with error bars (standard deviation)
        group_stats = df_clean.groupby(cat_var)[num_var].agg(['mean', 'std']).reset_index()
        fig = px.bar(group_stats, x=cat_var, y='mean', error_y='std',
                     labels={cat_var: cat_var, 'mean': f'Mean of {num_var}'},
                     title=f'Bar Plot of {num_var} by {cat_var} with Standard Deviation')
        st.plotly_chart(fig, use_container_width=True)

# =========================
#        STREAMLIT APP
# =========================

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="TextInsight Analyzer", layout="wide")
    st.title("📊 TextInsight Analyzer")
    
    # Disclaimer
    st.markdown("""
    ---
    <span style="font-size:0.9em; color:gray;">
    **Disclaimer:** This application is an amateur replication inspired by proprietary software for educational and research purposes only. 
    It is not affiliated with or endorsed by the creators of the original software. Performance and features may vary and are not comparable 
    to the official tool.
    </span>
    ---
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    st.sidebar.header("📥 Upload Files")
    uploaded_dataset = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])
    uploaded_wordlist = st.sidebar.file_uploader("Upload your wordlist (CSV, TXT, DIC, DICX, or Excel)", type=["csv", "txt", "dic", "dicx", "xls", "xlsx"])
    
    if uploaded_dataset and uploaded_wordlist:
        # Load dataset
        dataset = load_dataset(uploaded_dataset)
        
        # Load wordlist
        exact_words, wildcard_prefixes = load_wordlist(uploaded_wordlist)
        
        if dataset is not None and exact_words is not None and wildcard_prefixes is not None:
            # Display dataset preview
            st.subheader("📄 Dataset Preview")
            st.dataframe(dataset.head())
            
            # Display wordlist summary
            st.subheader("📃 Wordlist Summary")
            categories = list(exact_words.keys())
            st.write(f"**Categories:** {', '.join(categories)}")
            for category in categories:
                st.write(f"**{category}:**")
                st.write(f" - Exact Words: {len(exact_words[category])}")
                st.write(f" - Wildcard Prefixes: {len(wildcard_prefixes[category])}")
                # Show sample wordlist
                sample_exact = list(exact_words[category])[:5]
                sample_wildcard = wildcard_prefixes[category][:5]
                st.write(f"   - Sample Exact Words: {sample_exact if sample_exact else 'None'}")
                st.write(f"   - Sample Wildcard Prefixes: {sample_wildcard if sample_wildcard else 'None'}")
            
            # Select categories to analyze
            selected_categories = st.multiselect("📌 Select Categories to Analyze", options=categories, default=categories)
            
            if selected_categories:
                # Select text column
                text_columns = dataset.select_dtypes(include=['object', 'string']).columns.tolist()
                if not text_columns:
                    st.error("❌ No text columns found in the dataset. Please upload a dataset with at least one text column.")
                else:
                    text_column = st.selectbox("🔍 Select the column containing text data:", text_columns)
                    
                    # Start Analysis Button
                    if st.button("🚀 Start Analysis"):
                        with st.spinner("🔄 Performing Textual Analysis... This may take a while for large datasets."):
                            documents = dataset[text_column].astype(str)
                            progress_bar = st.progress(0)
                            # Filter wordlists based on selected categories
                            selected_exact_words = {cat: exact_words[cat] for cat in selected_categories}
                            selected_wildcard_prefixes = {cat: wildcard_prefixes[cat] for cat in selected_categories}
                            analysis_df = analyze_text(documents, selected_exact_words, selected_wildcard_prefixes, progress_bar)
                            enhanced_dataset = enhance_dataset(dataset, analysis_df)
                            st.success("✅ Textual Analysis Completed!")
                            
                            # Display enhanced dataset preview
                            st.subheader("📈 Enhanced Dataset Preview")
                            st.dataframe(enhanced_dataset.head())
                            
                            # Download enhanced dataset
                            csv = enhanced_dataset.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Enhanced Dataset (CSV)",
                                data=csv,
                                file_name="enhanced_dataset.csv",
                                mime="text/csv",
                            )
                            
                            # Generate and display word clouds for each category
                            for category in selected_categories:
                                column_name = f"{category}_detected_words"
                                if column_name in enhanced_dataset.columns:
                                    st.subheader(f"🌐 Word Cloud of Detected Words for '{category}'")
                                    generate_wordcloud(enhanced_dataset[column_name], category)
                            
                            # Divider before statistical analyses
                            st.markdown("---")
                            
                            # Statistical Analyses Section
                            st.subheader("📊 Statistical Analyses")
                            
                            # Create tabs for correlation and ANOVA
                            analysis_tab, anova_tab = st.tabs(["🔗 Pearson Correlation", "📉 ANOVA"])
                            
                            with analysis_tab:
                                perform_pearson_correlation(enhanced_dataset)
                            
                            with anova_tab:
                                perform_anova(enhanced_dataset)
            else:
                st.info("📌 Please select at least one category to analyze.")
    else:
        st.info("📌 Please upload both the dataset and the wordlist to begin.")
    
    # Footer
    st.markdown("""
    ---
    <span style="font-size:0.9em; color:gray;">
    **Note:** This tool is intended for educational and research purposes only. It is a simplified version and does not offer the comprehensive 
    features or performance of professional-grade software.
    </span>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
