"""Predict survival for a passenger using the trained model.

Usage:
    python src/predict.py            # runs a demo passenger
"""
from pathlib import Path

import joblib
import pandas as pd

import features  # noqa: F401  (needed so joblib can unpickle FeatureEngineer)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "titanic_model.joblib"


def predict(passenger: dict) -> tuple[int, float]:
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([passenger])
    return int(model.predict(df)[0]), float(model.predict_proba(df)[0, 1])


if __name__ == "__main__":
    passenger = {
        "Pclass": 1, "Name": "Doe, Mrs. Jane", "Sex": "female", "Age": 29,
        "SibSp": 1, "Parch": 0, "Fare": 80.0, "Cabin": "B22", "Embarked": "S",
    }
    label, prob = predict(passenger)
    print(f"Prediction: {'Survived' if label else 'Did not survive'} "
          f"(survival probability = {prob:.2%})")
