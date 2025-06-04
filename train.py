from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
import numpy as np
import joblib


RANDOM_STATE = 42


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    print("Тренируем модель..")
    pipeline.fit(X_train, y_train)
    print("Предсказываем...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Logistic Regression:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    return y_pred_proba


def threshold_optimizer(y_pred_proba, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Оптимальный порог: {optimal_threshold:.3f}")

    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    print(f"С оптимальным порогом: ({optimal_threshold:.3f}):")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_optimal):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_optimal):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
    return optimal_threshold


def train_model(data_path="data/toxic_comments.csv"):
    """Train the toxic comment classification model."""
    print("Starting model training...")

    # Load or create data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
    except FileNotFoundError:
        print("No training data found, creating sample data...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["toxic"],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"] if df["toxic"].nunique() > 1 else None,
    )

    pipeline_lg = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=10000,
                    stop_words="english",
                    min_df=2,
                    max_df=0.95,
                    strip_accents="ascii",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    penalty="l2",
                ),
            ),
        ]
    )

    # Train and evaluate model
    print("Training Logistic Regression model...")

    y_pred_proba = train_and_evaluate(pipeline_lg, X_train, y_train, X_test, y_test)
    # Optimize threshold
    print("Optimizing threshold...")
    optimal_threshold = threshold_optimizer(y_pred_proba, y_test)

    # Save model and threshold
    print("Saving model and threshold...")
    model_data = {"model": pipeline_lg, "threshold": optimal_threshold}
    joblib.dump(model_data, "models/latest_model.pkl")
    return True


if __name__ == "__main__":
    # Create directories
    import os

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_model()
