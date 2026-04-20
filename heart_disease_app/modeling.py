from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_disease_app.config import RANDOM_STATE, TEST_SIZE, get_model_candidates


@dataclass
class TrainingResult:
    leaderboard: pd.DataFrame
    fitted_models: Dict[str, Pipeline]
    feature_cols: List[str]
    best_model_name: str


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


@st.cache_resource
def train_models(df: pd.DataFrame, target_col: str) -> TrainingResult:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = []
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, base_model in get_model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC_AUC": _safe_roc_auc(y_test, y_score),
            }
        )
        fitted_models[model_name] = pipeline

    leaderboard = pd.DataFrame(rows).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    best_model_name = str(leaderboard.iloc[0]["Model"])

    return TrainingResult(
        leaderboard=leaderboard,
        fitted_models=fitted_models,
        feature_cols=feature_cols,
        best_model_name=best_model_name,
    )
