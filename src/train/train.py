"""Train the churn model and save it to disk."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"

DEFAULT_DATA = Path("data/churn_v1.csv")
DEFAULT_MODEL = Path("models/model.joblib")


def build_pipeline() -> Pipeline:
    """Preprocessing (scale numbers, one-hot text) + logistic regression."""
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])


def train(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL) -> dict:
    """Train on the CSV, save the model, and return the test metrics."""
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}")

    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    model = build_pipeline()
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "f1": round(f1_score(y_test, proba > 0.5), 3),
        "auc": round(roc_auc_score(y_test, proba), 3),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    metrics = train(args.data, args.model)
    print(f"Saved {args.model} | F1={metrics['f1']} AUC={metrics['auc']}")


if __name__ == "__main__":
    main()
