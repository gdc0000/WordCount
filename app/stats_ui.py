import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def perform_pearson_correlation(enhanced_df: pd.DataFrame) -> None:
    """
    Render Pearson correlation controls and output.
    """
    st.header("🔍 Pearson Correlation Analysis")

    numeric_cols = enhanced_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns available for correlation analysis.")
        return

    col1 = st.selectbox("Select the first numeric column:", options=numeric_cols, key="corr_col1")
    col2 = st.selectbox(
        "Select the second numeric column:", options=numeric_cols, index=1, key="corr_col2"
    )

    if st.button("Compute Pearson Correlation"):
        if col1 == col2:
            st.error("Please select two different columns.")
            return
        df_clean = enhanced_df[[col1, col2]].dropna()
        if df_clean.empty:
            st.error("No data available to compute correlation.")
            return
        corr_coef, p_value = pearsonr(df_clean[col1], df_clean[col2])
        st.write(f"**Pearson Correlation Coefficient:** {corr_coef:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")

        fig = px.scatter(
            df_clean,
            x=col1,
            y=col2,
            trendline="ols",
            title=f"Scatter Plot of {col1} vs {col2} with Trendline",
            labels={col1: col1, col2: col2},
        )
        st.plotly_chart(fig, use_container_width=True)


def perform_anova(enhanced_df: pd.DataFrame) -> None:
    """
    Render ANOVA controls and output.
    """
    st.header("📊 ANOVA (Analysis of Variance)")

    categorical_cols = enhanced_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = enhanced_df.select_dtypes(include=["number"]).columns.tolist()

    if not categorical_cols:
        st.warning("No categorical columns available for ANOVA.")
        return
    if not numeric_cols:
        st.warning("No numeric columns available for ANOVA.")
        return

    cat_var = st.selectbox(
        "Select the categorical (between factor) variable:", options=categorical_cols, key="anova_cat_var"
    )
    num_var = st.selectbox(
        "Select the numeric dependent variable:", options=numeric_cols, key="anova_num_var"
    )

    if st.button("Perform ANOVA"):
        df_clean = enhanced_df[[cat_var, num_var]].dropna()
        if df_clean.empty:
            st.error("No data available to perform ANOVA.")
            return

        df_clean[cat_var] = df_clean[cat_var].astype("category")

        try:
            model = ols(f"{num_var} ~ C({cat_var})", data=df_clean).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.write("**ANOVA Table:**")
            st.table(anova_table)

            if anova_table["PR(>F)"][0] < 0.05:
                st.success(
                    "The ANOVA test is significant (p < 0.05). There are significant differences between group means."
                )
            else:
                st.info(
                    "The ANOVA test is not significant (p ≥ 0.05). There are no significant differences between group means."
                )

            tukey = pairwise_tukeyhsd(endog=df_clean[num_var], groups=df_clean[cat_var], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

            st.write("**Tukey's HSD Post Hoc Comparisons:**")
            st.table(tukey_df)

            group_stats = df_clean.groupby(cat_var)[num_var].agg(["mean", "sem"]).reset_index()
            group_stats.rename(columns={"sem": "standard_error"}, inplace=True)
            group_stats["ci_lower"] = group_stats["mean"] - 1.96 * group_stats["standard_error"]
            group_stats["ci_upper"] = group_stats["mean"] + 1.96 * group_stats["standard_error"]

            fig = px.bar(
                group_stats,
                x="mean",
                y=cat_var,
                orientation="h",
                title=f"Bar Plot of {num_var} by {cat_var} with 95% Confidence Intervals",
                labels={"mean": f"Mean of {num_var}", cat_var: cat_var},
                height=600,
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})

            for i, row in group_stats.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row["ci_lower"],
                    y0=i,
                    x1=row["ci_upper"],
                    y1=i,
                    line=dict(color="black", width=2),
                )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Error performing ANOVA: {exc}")
