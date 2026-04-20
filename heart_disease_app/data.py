from typing import List, Tuple

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, target_col: str) -> Tuple[bool, str]:
    if df.empty:
        return False, "The dataset is empty. Please check the CSV file."
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' was not found in the dataset."

    target_values = sorted(df[target_col].dropna().unique().tolist())
    if not set(target_values).issubset({0, 1}):
        return False, "Target column must be a binary label (0/1)."

    return True, ""


def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    return [col for col in df.columns if col != target_col]


def get_input_statistics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    stats = pd.DataFrame(index=feature_cols)
    stats["median"] = df[feature_cols].median(numeric_only=True)
    stats["min"] = df[feature_cols].min(numeric_only=True)
    stats["max"] = df[feature_cols].max(numeric_only=True)
    return stats
