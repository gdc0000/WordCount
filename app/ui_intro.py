import streamlit as st


def render_intro() -> None:
    st.title("📊 WordCount Statistics")
    st.markdown(
        """
    ## 👋 Welcome to WordCount Statistics!

    **WordCount Statistics** is a tool designed to help you analyze textual data by leveraging customizable wordlists. Here's how to get started:

    ### 📁 Expected Files

    #### 1. **Dataset File**
    - **Format:** CSV or Excel (`.csv`, `.xls`, `.xlsx`)
    - **Requirements:**
        - Must contain at least one column with textual data.
        - Example columns:
            - `id`: Unique identifier for each entry.
            - `text`: The column containing the text you want to analyze.
    - **Sample Format:**

        | id | text                                     |
        |----|------------------------------------------|
        | 1  | I am happy and joyful today.             |
        | 2  | This is a sad and gloomy day.            |
        | ...| ...                                      |

    #### 2. **Wordlist (Dictionary) File**
    - **Format:** CSV, TXT, DIC, DICX, or Excel (`.csv`, `.txt`, `.dic`, `.dicx`, `.xls`, `.xlsx`)
    - **Requirements:**
        - Must contain a column named `DicTerm` with the words to analyze.
        - Additional columns represent categories. Mark a word for a category with an `X`.
        - **Wildcard Prefixes:** To indicate prefix matching, end a word with an asterisk (`*`). For example, `run*` will match `running`, `runner`, etc.
    - **Sample Format:**

        | DicTerm    | Intensifiers | Negations | Modal_Expressions |
        |------------|--------------|-----------|-------------------|
        | very       | X            |           |                   |
        | extremely  | X            |           |                   |
        | not        |              | X         |                   |
        | never      |              | X         |                   |
        | can*       |              |           | X                 |
        | might*     |              |           | X                 |

    ### 🛠️ Getting Started

    1. **Upload Files:**
        - Use the sidebar to upload your **Dataset** and **Wordlist** files.

    2. **Preview Data:**
        - After uploading, preview your dataset and review the wordlist summary.

    3. **Select Categories:**
        - Choose which categories from your wordlist you want to analyze.

    4. **Perform Analysis:**
        - Select the text column and start the analysis.

    5. **View Results:**
        - Explore the enhanced dataset, word frequency bar plots, and perform statistical analyses like Pearson Correlation and ANOVA.

    ### 📌 Notes
    - Ensure that your wordlist is properly formatted to achieve accurate analysis results.
    - The tool is intended for educational and research purposes. It may not cover all edge cases or complex textual nuances.
    """
    )
