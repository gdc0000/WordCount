# WordCount Statistics

**WordCount Statistics** is a Streamlit-based tool designed to analyze textual data using customizable wordlists. It not only counts the occurrences of specified words and phrases (including wildcard prefixes) in your text data but also provides detailed statistical insights and visualizations, such as word frequency bar plots, Pearson correlation, and ANOVA analyses.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

The **WordCount Statistics** tool allows you to:
- **Upload** a dataset (CSV, Excel) containing your text data.
- **Upload** a wordlist (CSV, TXT, DIC, DICX, or Excel) that defines words/phrases and their respective categories.
- **Analyze** the text for exact and wildcard (prefix) matches, both for single words and multi-word expressions.
- **Visualize** the results through interactive bar plots and conduct statistical analyses (Pearson correlation and ANOVA) to compare different metrics.

The application is especially useful for researchers and educators in the social sciences or digital humanities who need to perform a customizable textual analysis.

---

## Features

- **File Upload & Preview:** Upload your dataset and wordlist files, and preview them directly in the app.
- **Customizable Wordlists:** Define multiple categories in your wordlist. Use an `X` to mark terms for a category and append an asterisk (`*`) for prefix matching.
- **Text Tokenization:** The app tokenizes text into unigrams and generates n-grams (up to a configurable maximum).
- **Word Count Analysis:** Counts the occurrences of words/phrases per category and calculates the percentage relative to the total number of tokens.
- **Enhanced Dataset:** Combines original data with the analysis results and provides sanitized column names.
- **Interactive Visualizations:** Generates interactive Plotly bar plots for word frequency analysis.
- **Statistical Analysis:** Perform Pearson correlation between numeric columns and run ANOVA (with Tukey's HSD post hoc test) on selected variables.
- **Downloadable Results:** Download the enhanced dataset as a CSV file.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/wordcount-statistics.git
   cd wordcount-statistics


2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Required Dependencies**

   The required Python packages are listed in the [`requirements.txt`](./requirements.txt) file. Install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Streamlit App**

   Launch the application using:

   ```bash
   streamlit run wordcount.py
   ```

2. **Upload Your Files**

   - **Dataset File:** Upload a CSV or Excel file that contains at least one column with text data.
   - **Wordlist File:** Upload your dictionary file (CSV, TXT, DIC, DICX, or Excel) which must include a `DicTerm` column and additional category columns marked with an `X`.

3. **Configure and Analyze**

   - Preview your dataset and the wordlist summary.
   - Select the desired categories and the text column to analyze.
   - Click on **"Start Analysis"** to process your data.

4. **View Results and Visualizations**

   - Inspect the enhanced dataset with analysis results.
   - Generate interactive bar plots for word frequency.
   - Perform Pearson correlation and ANOVA analyses via the provided tabs.
   - Download the enhanced dataset as a CSV file.

---

## File Structure

```
.
├── requirements.txt        # Lists required Python packages.
├── wordcount.py            # Main Streamlit application code.
└── README.md               # This file.
```

---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Open a pull request describing your changes.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contact

**Gabriele Di Cicco, PhD in Social Psychology**  
[GitHub](https://github.com/gdc0000) | [ORCID](https://orcid.org/0000-0002-1439-5790) | [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)

---

Happy Analyzing!
```
