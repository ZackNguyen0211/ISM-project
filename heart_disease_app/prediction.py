from typing import Dict, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_probability(model: Pipeline, input_data: Dict[str, float]) -> Tuple[float, int]:
    user_df = pd.DataFrame([input_data])
    pred_label = int(model.predict(user_df)[0])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(user_df)[0, 1])
    else:
        score = float(model.decision_function(user_df)[0])
        prob = 1.0 / (1.0 + pow(2.718281828, -score))

    return prob, pred_label
