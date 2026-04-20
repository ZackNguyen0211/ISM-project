# Heart Disease Risk Prediction Web App

The app applies Data Mining techniques to predict the **percentage risk of heart disease for one person**.

## Project Objectives

- Predict heart disease probability from patient health indicators.
- Compare multiple classification models and automatically select the best one by ROC-AUC.
- Provide a web interface where users enter one patient profile and click Predict.

## Core Features

- Load and validate dataset from `heart.csv`.
- Train and evaluate 4 models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors
- Display a model leaderboard using Accuracy, Precision, Recall, F1, and ROC-AUC.
- Select the best model automatically for prediction.
- Provide a patient input form and return:
  - Heart disease probability (%)
  - Risk assessment label (high risk / low risk)

## Project Structure

- `app.py`: application entry point and flow orchestration
- `heart_disease_app/config.py`: app settings and model candidates
- `heart_disease_app/data.py`: data loading, validation, and input statistics
- `heart_disease_app/modeling.py`: model training, evaluation, and best-model selection
- `heart_disease_app/prediction.py`: probability prediction logic
- `heart_disease_app/ui.py`: Streamlit UI components
- `heart.csv`: source dataset
- `requirements.txt`: required dependencies

## Run the App

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the web app:

```bash
streamlit run app.py
```

3. Open the local Streamlit URL (default: `http://localhost:8501`).

## Usage Flow

1. Open **Model Training** to train all models.
2. Review the leaderboard and selected best model.
3. Open **Predict Risk %** and enter one patient profile.
4. Click **Predict** to get the heart disease probability.

## Team Work Split (6 Members)

### Member 1 - Data Engineer
- Owns `heart_disease_app/data.py`
- Responsibilities:
  - Handle dataset input and loading
  - Validate binary target labels (0/1)
  - Compute median/min/max values for prediction form defaults

### Member 2 - ML Engineer (Model Training)
- Owns `heart_disease_app/modeling.py`
- Responsibilities:
  - Build preprocessing + model pipelines
  - Train 4 classification models
  - Implement leaderboard and best-model selection logic

### Member 3 - Prediction Engineer
- Owns `heart_disease_app/prediction.py`
- Responsibilities:
  - Process one-patient input data
  - Calculate probability predictions
  - Convert outputs to risk percentage and classification label

### Member 4 - Frontend/App Engineer
- Owns `heart_disease_app/ui.py`
- Responsibilities:
  - Build Streamlit UI components
  - Design patient input form
  - Display metrics, charts, and prediction results

### Member 5 - Integrator/PM
- Owns `app.py` and project documentation
- Responsibilities:
  - Orchestrate flow across all modules
  - Manage session state and app lifecycle
  - Validate end-to-end usage flow
  - Maintain README and project run instructions

### Member 6 - QA/Release Engineer
- Owns testing, quality gates, and release readiness
- Responsibilities:
  - Create manual and automated test scenarios for data validation, training flow, and prediction outputs
  - Run regression checks whenever model settings or UI behavior change
  - Validate probability output consistency across repeated runs
  - Maintain release checklist, known issues list, and versioned delivery notes

## Notes

- Default target column: `target`
- Default CSV file: `heart.csv`
- You can change CSV path and target column directly from the sidebar.
