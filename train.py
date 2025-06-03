import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import re
from datetime import datetime


def preprocess_text(text):
    """Basic text preprocessing."""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def train_model(data_path="data/toxic_comments.csv"):
    """Train the toxic comment classification model."""
    print("Starting model training...")

    # Load or create data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
    except FileNotFoundError:
        print("No training data found, creating sample data...")

    # Preprocess text
    df["text_clean"] = df["text"].apply(preprocess_text)

    # Remove empty texts
    df = df[df["text_clean"].str.len() > 0]

    if len(df) < 10:
        print("Not enough data for training. Need at least 10 samples.")
        return False

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["toxic"],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"] if df["toxic"].nunique() > 1 else None,
    )

    # Vectorize text (TF-IDF)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )

    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision (Toxic): {precision:.3f}")
    print(f"Recall (Toxic): {recall:.3f}")
    print(f"F1-Score (Toxic): {f1:.3f}")

    # Save model and vectorizer
    model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pickle.dump(model, open(f"models/{model_version}.pkl", "wb"))
    pickle.dump(vectorizer, open(f"models/{model_version}_vectorizer.pkl", "wb"))

    # Save latest model (for easy loading)
    pickle.dump(model, open("models/latest_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/latest_vectorizer.pkl", "wb"))

    # Save performance metrics
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    print(f"Model saved as {model_version}")
    print(f"Performance metrics: {metrics}")
    return True


if __name__ == "__main__":
    # Create directories
    import os

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    train_model()
