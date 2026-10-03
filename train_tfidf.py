import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Text cleaning function: remove HTML tags and convert to lowercase
def clean_text(text):
    # Remove HTML tags such as <br />, <br>
    text = re.sub(r"<[^>]+>", " ", str(text))
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Convert to lowercase and trim whitespace
    return text.lower().strip()


def main():
    print("Loading IMDB Dataset...")
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"Loaded {len(df)} reviews.")

    # Clean the review column
    print("Cleaning reviews (removing HTML tags and converting to lowercase)...")
    df["clean_review"] = df["review"].apply(clean_text)

    # Split into 80% training and 20% testing with random_state=42
    print("Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create TF-IDF + Logistic Regression pipeline
    print("Training TF-IDF + Logistic Regression model...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=25000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the pipeline on the test set
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model with joblib
    model_filename = "tfidf_model.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Model successfully saved to '{model_filename}'")


if __name__ == "__main__":
    main()
