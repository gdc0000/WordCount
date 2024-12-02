import streamlit as st
import pandas as pd
import re
import time

# Define functions
def load_dataset(uploaded_file):
    """Load a dataset from a CSV or Excel file."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")

def load_wordlist(uploaded_file):
    """Load a wordlist from a CSV file."""
    wordlist_df = pd.read_csv(uploaded_file)
    return wordlist_df

def count_lexicon_regex_asterisk(document, word_list):
    """Perform word count analysis using regex with optional wildcards."""
    clean_document = re.sub(r"[^\w\s']", ' ', document.lower())
    tokens = clean_document.split()

    regex_patterns = []
    for word in word_list:
        if word.endswith('*'):
            pattern = r'\b' + re.escape(word[:-1]) + r'\w*'
        else:
            pattern = r'\b' + re.escape(word) + r'\b'
        regex_patterns.append(pattern)

    combined_pattern = '|'.join(regex_patterns)
    detected_words = re.findall(combined_pattern, clean_document)

    count = len(detected_words)
    len_document = len(document)
    n_tokens = len(tokens)
    n_types = len(set(tokens))
    word_perc = count / n_tokens if n_tokens > 0 else 0

    return clean_document, len_document, n_tokens, n_types, count, word_perc, detected_words

def analyze_text(documents, wordlist, progress_bar):
    """Perform textual analysis on documents using a wordlist."""
    results = []
    total_docs = len(documents)
    for i, doc in enumerate(documents):
        results.append(count_lexicon_regex_asterisk(doc, wordlist))
        progress_bar.progress((i + 1) / total_docs)  # Update progress bar
    analysis_df = pd.DataFrame(results, columns=[
        'clean_document', 'len_document', 'n_tokens', 'n_types', 'count', 'word_perc', 'detected_words'
    ])
    return analysis_df

def enhance_dataset(dataset, text_column, analysis_df, label):
    """Add analysis results to the original dataset."""
    analysis_df = analysis_df.rename(columns={
        'count': f'{label}_count',
        'word_perc': f'{label}_word_perc'
    })
    dataset = pd.concat([dataset.reset_index(drop=True), analysis_df[[f'{label}_count', f'{label}_word_perc']]], axis=1)
    return dataset

# Streamlit App
st.title("Textual Analysis App")

# File upload section
st.sidebar.header("Upload Files")
uploaded_dataset = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])
uploaded_wordlist = st.sidebar.file_uploader("Upload your wordlist (CSV)", type=["csv"])

if uploaded_dataset and uploaded_wordlist:
    try:
        # Load files
        dataset = load_dataset(uploaded_dataset)
        wordlist_df = load_wordlist(uploaded_wordlist)
        
        # Validate wordlist
        if 'word' not in wordlist_df.columns:
            st.error("The wordlist CSV must contain a column named 'word'.")
        else:
            wordlist = wordlist_df['word'].str.lower().tolist()

            st.write("### Dataset Preview")
            st.dataframe(dataset.head())

            st.write("### Wordlist Preview")
            st.dataframe(wordlist_df.head())

            # User input for label
            label = st.text_input("Enter the label/category name for this wordlist (e.g., 'Emotions'):").strip()

            if label:
                # Select text column
                text_column = st.selectbox("Select the column containing text data:", dataset.columns)
                documents = dataset[text_column].astype(str)

                if st.button("Start Analysis"):
                    st.write("### Performing Textual Analysis...")
                    progress_bar = st.progress(0)  # Initialize progress bar
                    analysis_df = analyze_text(documents, wordlist, progress_bar)

                    # Enhance dataset
                    enhanced_dataset = enhance_dataset(dataset, text_column, analysis_df, label)
                    st.write("### Enhanced Dataset")
                    st.dataframe(enhanced_dataset.head())

                    # Download enhanced dataset
                    output_csv = enhanced_dataset.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Enhanced Dataset (CSV)",
                        data=output_csv,
                        file_name="enhanced_dataset.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Please enter a label for the wordlist.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both the dataset and the wordlist to begin.")
