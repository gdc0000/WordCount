import streamlit as st
import pandas as pd
import re
from typing import List, Tuple

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
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_wordlist(uploaded_file):
    """
    Load and preprocess a wordlist from a CSV file.
    
    Parameters:
        uploaded_file: Uploaded file object from Streamlit.
        
    Returns:
        Tuple containing:
            - Set of exact words
            - List of wildcard prefixes
    """
    try:
        wordlist_df = pd.read_csv(uploaded_file)
        if 'word' not in wordlist_df.columns:
            st.error("The wordlist CSV must contain a column named 'word'.")
            return None, None
        # Drop NaN and convert to lowercase
        wordlist = wordlist_df['word'].dropna().astype(str).str.lower().tolist()
        # Split into exact words and wildcard prefixes
        exact_words = set()
        wildcard_prefixes = []
        for word in wordlist:
            if word.endswith('*'):
                prefix = word[:-1]
                if prefix:  # Avoid empty prefix
                    wildcard_prefixes.append(prefix)
            else:
                exact_words.add(word)
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

def process_document(doc: str, exact_words: set, wildcard_prefixes: List[str]) -> Tuple[int, int, int, float, List[str]]:
    """
    Process a single document and return analysis metrics.
    
    Parameters:
        doc: The text document as a string.
        exact_words: Set of exact words to match.
        wildcard_prefixes: List of prefixes for wildcard matching.
        
    Returns:
        Tuple containing:
            - Number of tokens
            - Number of types
            - Count of wordlist identified
            - Proportion of wordlist words per token
            - List of detected words
    """
    tokens = clean_and_tokenize(doc)
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    
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
    
    word_perc = count / n_tokens if n_tokens > 0 else 0.0
    detected_words_list = list(detected_words)
    
    return n_tokens, n_types, count, word_perc, detected_words_list

def analyze_text(documents: pd.Series, exact_words: set, wildcard_prefixes: List[str], progress_bar) -> pd.DataFrame:
    """
    Analyze text in documents using the wordlist.
    
    Parameters:
        documents: pandas Series containing text data.
        exact_words: Set of exact words to match.
        wildcard_prefixes: List of prefixes for wildcard matching.
        progress_bar: Streamlit progress bar object.
        
    Returns:
        pandas.DataFrame with analysis results.
    """
    analysis_results = []
    total_docs = len(documents)
    for i, doc in enumerate(documents):
        if pd.isna(doc):
            doc = ""
        n_tokens, n_types, count, word_perc, detected_words = process_document(doc, exact_words, wildcard_prefixes)
        analysis_results.append({
            'n_tokens': n_tokens,
            'n_types': n_types,
            'word_count': count,
            'word_perc': word_perc,
            'detected_words': detected_words
        })
        # Update progress every 100 documents or at the end
        if (i + 1) % 100 == 0 or i == total_docs -1:
            progress_bar.progress((i + 1) / total_docs)
    analysis_df = pd.DataFrame(analysis_results)
    return analysis_df

def enhance_dataset(dataset: pd.DataFrame, analysis_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Enhance the original dataset with analysis results.
    
    Parameters:
        dataset: Original pandas DataFrame.
        analysis_df: DataFrame containing analysis results.
        label: Label/category name for the wordlist.
        
    Returns:
        Enhanced pandas DataFrame.
    """
    dataset = dataset.reset_index(drop=True)
    analysis_df = analysis_df.rename(columns={
        'n_tokens': f'{label}_n_tokens',
        'n_types': f'{label}_n_types',
        'word_count': f'{label}_word_count',
        'word_perc': f'{label}_word_perc',
        'detected_words': f'{label}_detected_words'
    })
    enhanced_dataset = pd.concat([dataset, analysis_df], axis=1)
    return enhanced_dataset

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
    uploaded_wordlist = st.sidebar.file_uploader("Upload your wordlist (CSV)", type=["csv"])
    
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
            st.write(f"**Exact Words:** {len(exact_words)}")
            st.write(f"**Wildcard Prefixes:** {len(wildcard_prefixes)}")
            # Show sample wordlist
            sample_exact = list(exact_words)[:5]
            sample_wildcard = wildcard_prefixes[:5]
            st.write("**Sample Exact Words:**", sample_exact)
            st.write("**Sample Wildcard Prefixes:**", sample_wildcard)
            
            # User input for label
            label = st.text_input("📝 Enter the label/category name for this wordlist (e.g., 'Emotions')").strip()
            
            if label:
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
                            analysis_df = analyze_text(documents, exact_words, wildcard_prefixes, progress_bar)
                            enhanced_dataset = enhance_dataset(dataset, analysis_df, label)
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
            else:
                st.info("📝 Please enter a label for the wordlist.")
    else:
        st.info("📌 Please upload both the dataset and the wordlist to begin.")

    # Footer with disclaimer
    st.markdown("""
    ---
    **Note:** This tool is intended for educational and research purposes only. It is a simplified version and does not offer the comprehensive features or performance of professional-grade software.
    """)
    
if __name__ == "__main__":
    main()
