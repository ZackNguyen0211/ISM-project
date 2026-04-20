import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(page_title="Heart Disease Data Mining App", layout="wide")


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_models(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, Pipeline], list]:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    numeric_features = list(X.columns)
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
                numeric_features,
            )
        ]
    )

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fitted_models: Dict[str, Pipeline] = {}
    rows = []

    for name, base_model in model_defs.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model),
            ]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            y_score = pipe.decision_function(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC_AUC": _safe_roc_auc(y_test, y_score),
            }
        )
        fitted_models[name] = pipe

    leaderboard = pd.DataFrame(rows).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    return leaderboard, fitted_models, feature_cols


@st.cache_data
def run_clustering(df: pd.DataFrame, feature_cols: list, k: int):
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    silhouette = silhouette_score(X_scaled, clusters) if k > 1 else float("nan")

    clustered_df = df.copy()
    clustered_df["cluster"] = clusters

    profile = clustered_df.groupby("cluster")[feature_cols].mean().round(2)
    return coords, clusters, silhouette, profile


def render_overview(df: pd.DataFrame, target_col: str):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.write("Sample data")
    st.dataframe(df.head(15), use_container_width=True)

    st.write("Missing values")
    st.dataframe(df.isna().sum().to_frame("missing_count"), use_container_width=True)

    if target_col in df.columns:
        st.write("Target distribution")
        target_dist = df[target_col].value_counts().sort_index()
        st.bar_chart(target_dist)


def render_modeling(df: pd.DataFrame, target_col: str):
    st.subheader("Supervised Data Mining: Classification")
    st.write("Compares multiple classifiers and selects the best one by ROC-AUC.")

    leaderboard, fitted_models, feature_cols = train_models(df, target_col)
    st.dataframe(leaderboard.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "ROC_AUC": "{:.3f}",
    }), use_container_width=True)

    best_model_name = leaderboard.iloc[0]["Model"]
    st.success(f"Best model by ROC-AUC: {best_model_name}")

    st.session_state["fitted_models"] = fitted_models
    st.session_state["best_model_name"] = best_model_name
    st.session_state["feature_cols"] = feature_cols

    best_model = fitted_models[best_model_name]
    model_obj = best_model.named_steps["model"]

    st.write("Feature impact snapshot")
    if hasattr(model_obj, "feature_importances_"):
        importances = pd.Series(model_obj.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.bar_chart(importances)
    elif hasattr(model_obj, "coef_"):
        coefs = pd.Series(np.abs(model_obj.coef_.ravel()), index=feature_cols).sort_values(ascending=False)
        st.bar_chart(coefs)
    else:
        st.info("This model does not expose feature importances directly.")


def render_clustering(df: pd.DataFrame, target_col: str):
    st.subheader("Unsupervised Data Mining: Clustering")
    feature_cols = [c for c in df.columns if c != target_col]
    k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)

    coords, clusters, silhouette, profile = run_clustering(df, feature_cols, k)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="viridis", alpha=0.8)
        ax.set_title("K-Means clusters projected with PCA")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
        ax.add_artist(legend)
        st.pyplot(fig)

    with c2:
        st.metric("Silhouette Score", f"{silhouette:.3f}")
        st.caption("Higher values usually indicate better cluster separation.")

    st.write("Cluster profile (mean feature values)")
    st.dataframe(profile, use_container_width=True)


def render_prediction(df: pd.DataFrame, target_col: str):
    st.subheader("Prediction")
    if "fitted_models" not in st.session_state:
        st.warning("Run the Classification tab first so the app can train models.")
        return

    best_model_name = st.session_state["best_model_name"]
    model = st.session_state["fitted_models"][best_model_name]
    feature_cols = st.session_state["feature_cols"]

    st.write(f"Using best model: {best_model_name}")

    medians = df[feature_cols].median(numeric_only=True)
    mins = df[feature_cols].min(numeric_only=True)
    maxs = df[feature_cols].max(numeric_only=True)

    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        for idx, col in enumerate(feature_cols):
            with cols[idx % 3]:
                input_data[col] = st.number_input(
                    label=col,
                    value=float(medians[col]),
                    min_value=float(mins[col]),
                    max_value=float(maxs[col]),
                    step=0.1,
                )
        submitted = st.form_submit_button("Predict")

    if submitted:
        pred_df = pd.DataFrame([input_data])
        pred = int(model.predict(pred_df)[0])

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(pred_df)[0, 1])
            st.metric("Risk probability", f"{prob:.2%}")

        label = "Higher heart disease risk" if pred == 1 else "Lower heart disease risk"
        st.success(f"Prediction: {label}")


def main():
    st.title("Heart Disease Data Mining Application")
    st.write(
        "This app demonstrates core data mining techniques on your dataset: "
        "classification, clustering, and interactive prediction."
    )

    default_path = "heart.csv"
    data_path = st.sidebar.text_input("CSV path", value=default_path)

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    df = load_data(data_path)

    target_col = st.sidebar.text_input("Target column", value="target")
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' is not present in dataset.")
        st.stop()

    tabs = st.tabs(["Overview", "Classification", "Clustering", "Prediction"])

    with tabs[0]:
        render_overview(df, target_col)
    with tabs[1]:
        render_modeling(df, target_col)
    with tabs[2]:
        render_clustering(df, target_col)
    with tabs[3]:
        render_prediction(df, target_col)


if __name__ == "__main__":
    main()
