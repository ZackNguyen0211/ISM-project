from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

APP_TITLE = "Heart Disease Risk Prediction Web App"
APP_DESCRIPTION = (
    "A Data Mining web application that predicts the percentage risk of heart disease."
)
DEFAULT_DATA_PATH = "heart.csv"
DEFAULT_TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_model_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.7,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    }
