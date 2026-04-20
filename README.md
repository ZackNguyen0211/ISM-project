# Heart Disease Data Mining Application

This project is a complete Data Mining application built with Python and Streamlit using the provided `heart.csv` dataset.

## Data Mining Techniques Used

- Supervised learning (classification)
- Multi-model comparison (Logistic Regression, Random Forest, Gradient Boosting, KNN)
- Model evaluation with Accuracy, Precision, Recall, F1, ROC-AUC
- Unsupervised learning (K-Means clustering)
- PCA-based cluster visualization
- Interactive risk prediction from user inputs

## Project Structure

- `heart.csv` - source dataset
- `app.py` - Streamlit application
- `requirements.txt` - required Python packages

## Run the Application

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Notes

- Default target column is `target`.
- Default data path is `heart.csv`.
- In the app, visit the **Classification** tab first to train models, then use the **Prediction** tab.
