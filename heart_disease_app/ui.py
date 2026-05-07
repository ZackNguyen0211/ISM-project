from typing import Dict, List

import pandas as pd
import streamlit as st


def render_header(title: str, description: str):
    st.title(title)
    st.caption(description)


def render_dataset_overview(df: pd.DataFrame, target_col: str):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate rows", int(df.duplicated().sum()))

    st.write("Sample data")
    st.dataframe(df.head(15), use_container_width=True)

    st.write("Missing values")
    st.dataframe(df.isna().sum().to_frame("missing_count"), use_container_width=True)

    st.write("Target distribution")
    st.bar_chart(df[target_col].value_counts().sort_index())


def render_leaderboard(leaderboard: pd.DataFrame):
    st.subheader("Model leaderboard")
    st.dataframe(
        leaderboard.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC_AUC": "{:.3f}",
                "CV_ROC_AUC": "{:.3f}",
                "CV_Std": "{:.6f}",
            }
        ),
        use_container_width=True,
    )
    
    # Check for overfitting (gap between test ROC-AUC and CV ROC-AUC)
    best_row = leaderboard.iloc[0]
    roc_auc = best_row.get("ROC_AUC", 0)
    cv_roc_auc = best_row.get("CV_ROC_AUC", 0)
    gap = roc_auc - cv_roc_auc if cv_roc_auc > 0 else 0
    
    if gap > 0.05:  # Large gap indicates overfitting
        st.warning(
            f"⚠️ **{best_row['Model']}** shows signs of overfitting! "
            f"Test ROC-AUC ({roc_auc:.3f}) >> CV ROC-AUC ({cv_roc_auc:.3f}). "
            f"Consider using a different model or adjusting hyperparameters."
        )
    elif roc_auc >= 0.99 and cv_roc_auc >= 0.99:
        st.warning(
            f"⚠️ **{best_row['Model']}** has perfect/near-perfect accuracy. "
            f"This may indicate data leakage or overfitting. Verify data quality and splits."
        )


def render_feature_impact(model_name: str, model, feature_cols: List[str]):
    st.subheader("Feature impact")
    st.caption(f"Current model: {model_name}")

    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.bar_chart(importances)
    elif hasattr(estimator, "coef_"):
        coefs = pd.Series(estimator.coef_.ravel(), index=feature_cols).abs().sort_values(ascending=False)
        st.bar_chart(coefs)
    else:
        st.info("This model does not provide direct feature importance.")


def render_prediction_form(feature_cols: List[str], input_stats: pd.DataFrame) -> Dict[str, float]:
    st.subheader("Patient input")
    st.write("Enter patient indicators and click Predict to estimate heart disease risk percentage.")

    user_input: Dict[str, float] = {}
    with st.form("prediction_form"):
        columns = st.columns(3)
        for idx, feature in enumerate(feature_cols):
            with columns[idx % 3]:
                user_input[feature] = st.number_input(
                    label=feature,
                    value=float(input_stats.loc[feature, "median"]),
                    min_value=float(input_stats.loc[feature, "min"]),
                    max_value=float(input_stats.loc[feature, "max"]),
                    step=0.1,
                )
        submitted = st.form_submit_button("Predict")

    if submitted:
        return user_input
    return {}


def render_prediction_result(probability: float, label: int):
    st.subheader("Prediction result")
    st.metric("Heart disease probability", f"{probability:.2%}")

    if label == 1:
        st.error("Assessment: High-risk profile (predicted class = 1).")
    else:
        st.success("Assessment: Low-risk profile (predicted class = 0).")
