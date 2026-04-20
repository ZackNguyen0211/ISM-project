import os

import streamlit as st

from heart_disease_app.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET_COLUMN,
)
from heart_disease_app.data import (
    get_input_statistics,
    validate_dataset,
    load_data,
)
from heart_disease_app.modeling import train_models
from heart_disease_app.prediction import predict_probability
from heart_disease_app.ui import (
    render_dataset_overview,
    render_feature_impact,
    render_header,
    render_leaderboard,
    render_prediction_form,
    render_prediction_result,
)


st.set_page_config(page_title=APP_TITLE, layout="wide")


def main():
    render_header(APP_TITLE, APP_DESCRIPTION)

    data_path = st.sidebar.text_input("CSV path", value=DEFAULT_DATA_PATH)
    target_col = st.sidebar.text_input("Target column", value=DEFAULT_TARGET_COLUMN)

    if not os.path.exists(data_path):
        st.error(f"Dataset not found at: {data_path}")
        st.stop()

    df = load_data(data_path)
    is_valid, error_message = validate_dataset(df, target_col)
    if not is_valid:
        st.error(error_message)
        st.stop()

    tabs = st.tabs(["Overview", "Model Training", "Predict Risk %"])

    with tabs[0]:
        render_dataset_overview(df, target_col)

    with tabs[1]:
        st.subheader("Data Mining Approach")
        st.write("The system trains 4 classification models and selects the best one by ROC-AUC.")

        training_result = train_models(df, target_col)
        render_leaderboard(training_result.leaderboard)
        st.success(f"Best model by ROC-AUC: {training_result.best_model_name}")

        st.session_state["fitted_models"] = training_result.fitted_models
        st.session_state["best_model_name"] = training_result.best_model_name
        st.session_state["feature_cols"] = training_result.feature_cols

        best_model = training_result.fitted_models[training_result.best_model_name]
        render_feature_impact(training_result.best_model_name, best_model, training_result.feature_cols)

    with tabs[2]:
        if "fitted_models" not in st.session_state:
            st.warning("Please open the Model Training tab first to train the models.")
            st.stop()

        best_model_name = st.session_state["best_model_name"]
        best_model = st.session_state["fitted_models"][best_model_name]
        feature_cols = st.session_state["feature_cols"]

        input_stats = get_input_statistics(df, feature_cols)
        user_input = render_prediction_form(feature_cols, input_stats)
        if user_input:
            probability, label = predict_probability(best_model, user_input)
            render_prediction_result(probability, label)


if __name__ == "__main__":
    main()
