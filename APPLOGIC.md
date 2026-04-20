# Application Logic and Background Algorithms

## 1) Project Purpose
This application predicts the percentage risk of heart disease for one patient based on clinical indicators.
It is implemented as a Streamlit web app and uses supervised machine learning for binary classification (0 or 1).

## 2) End-to-End Runtime Flow
1. The app starts and loads global configuration such as default CSV path, target column, and model settings from "heart_disease_app\config.py".
2. The user provides in "app.py":
   - CSV file path
   - Target column name
3. The dataset is loaded and validated in "heart_disease_app\data.py":
   - Dataset must not be empty
   - Target column must exist
   - Target values must be binary (0/1)
4. In Model Training, the system trains 4 models in "heart_disease_app\modeling.py" and evaluates them.
5. The app selects the best model using ROC-AUC as the primary ranking metric.
6. In Predict Risk %, the user enters one patient profile and application will use the best model above to predict in "heart_disease_app\prediction.py".
7. The selected best model returns in "heart_disease_app\ui.py":
   - Predicted class (high-risk or low-risk profile)
   - Probability of class 1, shown as a percentage

## 3) Algorithms Applied in the Background
The project trains and compares 4 classification algorithms:

1. Logistic Regression
- Linear model for binary classification
- Produces calibrated probability via predict_proba

2. Random Forest
- Ensemble of decision trees using bagging
- Strong baseline for tabular clinical data

3. Gradient Boosting
- Sequential boosting ensemble that corrects prior errors
- Often strong on structured datasets

4. K-Nearest Neighbors (KNN)
- Distance-based non-parametric method
- Sensitive to feature scale, so standardization is important

## 4) Data Processing and Training Pipeline
Each model is trained through a consistent preprocessing + modeling pipeline:
- Missing value handling: median imputation
- Feature scaling: StandardScaler normalization
- Data split: train/test split at 80/20
- Stratification: enabled to preserve class distribution
- Reproducibility: fixed random state

## 5) Model Evaluation and Selection Strategy
For each candidate model, the app computes:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Selection rule:
- Best model = model with the highest ROC-AUC on the test split

Why ROC-AUC is prioritized:
- It evaluates ranking quality across decision thresholds
- It is more informative than raw accuracy in many binary-risk settings

## 6) Prediction Logic for Risk Percentage
During inference for one patient:
- If the model supports predict_proba, the app uses probability of class 1 directly.
- If predict_proba is not available, the app uses decision_function score and converts it with a sigmoid function.
- Output is displayed as a percentage probability and a risk label.

## 7) Architecture by Module
app.py
- Main app orchestration
- Tab navigation
- Session state management

heart_disease_app/config.py
- Global configuration constants
- Candidate model definitions

heart_disease_app/data.py
- Data loading
- Dataset validation
- Input statistics for UI defaults

heart_disease_app/modeling.py
- Preprocessing pipeline
- Model training and evaluation
- Leaderboard generation and best-model selection

heart_disease_app/prediction.py
- Single-patient inference logic
- Probability extraction and fallback conversion

heart_disease_app/ui.py
- UI rendering for overview, training view, input form, and prediction output

## 8) Why This Design Is Practical for Team Projects
- Clear separation of concerns across modules
- Easy parallel development by multiple team members
- Consistent training logic across all model candidates
- Transparent model comparison and explainable selection rule

## 9) Updated Task Allocation for a 6-Person Team

### Member 1 - Data Engineer
- Owns data loading and validation logic
- Handles input schema checks and data integrity safeguards

### Member 2 - ML Engineer (Training)
- Owns preprocessing and model training pipelines
- Tunes model settings and monitors training metrics

### Member 3 - Prediction Engineer
- Owns inference path for single-patient prediction
- Maintains probability extraction and fallback conversion logic

### Member 4 - Frontend/App Engineer
- Owns user interface and interaction flow
- Maintains patient input form and result visualization components

### Member 5 - Integrator/PM
- Owns cross-module integration and delivery coordination
- Maintains app orchestration and documentation consistency

### Member 6 - QA/Release Engineer (New Workstream)
- Owns quality assurance and release readiness
- Defines test matrix for data validation, training, and prediction behavior
- Runs regression checks before each release
- Maintains release checklist, known issues, and handover notes

This additional QA/Release stream is intentionally introduced to ensure the workload scales cleanly from 5 to 6 contributors.
