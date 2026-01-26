import pandas as pd


def enhance_dataset(dataset: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance the original dataset with analysis results and sanitize column names.
    """
    dataset = dataset.reset_index(drop=True)
    enhanced_dataset = pd.concat([dataset, analysis_df], axis=1)

    list_columns = enhanced_dataset.columns[
        enhanced_dataset.applymap(lambda x: isinstance(x, list)).any()
    ]
    dict_columns = enhanced_dataset.columns[
        enhanced_dataset.applymap(lambda x: isinstance(x, dict)).any()
    ]
    set_columns = enhanced_dataset.columns[
        enhanced_dataset.applymap(lambda x: isinstance(x, set)).any()
    ]

    def convert_to_string(value):
        if isinstance(value, list):
            return ", ".join(map(str, value))
        if isinstance(value, dict):
            return ", ".join([f"{k}: {v}" for k, v in value.items()])
        if isinstance(value, set):
            return ", ".join(map(str, value))
        return value

    for col in list_columns.tolist() + dict_columns.tolist() + set_columns.tolist():
        enhanced_dataset[col] = enhanced_dataset[col].apply(convert_to_string)

    object_columns = enhanced_dataset.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        enhanced_dataset[col] = enhanced_dataset[col].astype(str)

    enhanced_dataset.columns = (
        enhanced_dataset.columns.str.replace(" ", "_").str.replace("[^A-Za-z0-9_]", "", regex=True)
    )

    if "n_tokens" not in enhanced_dataset.columns:
        enhanced_dataset["n_tokens"] = analysis_df["n_tokens"]
    if "n_types" not in enhanced_dataset.columns:
        enhanced_dataset["n_types"] = analysis_df["n_types"]

    return enhanced_dataset
