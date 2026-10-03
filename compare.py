import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline

TEST_SUBSET_SIZE = 1000
TFIDF_MODEL_PATH = "tfidf_model.joblib"
TRANSFORMER_MODEL_DIR = "transformer_model"


# Text cleaning function: remove HTML tags and lowercase
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def main():
    # Verify that model files exist before evaluating
    if not os.path.exists(TFIDF_MODEL_PATH):
        print(f"Error: TF-IDF model not found at '{TFIDF_MODEL_PATH}'.")
        print("Please train it first: python train_tfidf.py")
        return

    if not os.path.exists(TRANSFORMER_MODEL_DIR):
        print(f"Error: Transformer model not found at '{TRANSFORMER_MODEL_DIR}'.")
        print("Please fine-tune it first: python train_transformer.py")
        return

    print("Loading IMDB Dataset...")
    df = pd.read_csv("IMDB Dataset.csv")

    # Clean reviews
    print("Cleaning review text...")
    df["review"] = df["review"].apply(clean_text)

    # 80/20 train/test split with random_state=42 (identical across all scripts)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Select the exact same 1,000 test reviews
    eval_df = test_df.iloc[:TEST_SUBSET_SIZE]
    y_true = eval_df["sentiment"].values
    reviews = eval_df["review"].tolist()
    print(f"Evaluating both models on the same {len(eval_df)} test reviews...\n")

    # 1. Evaluate TF-IDF + Logistic Regression
    print("1. Evaluating TF-IDF + Logistic Regression model...")
    tfidf_model = joblib.load(TFIDF_MODEL_PATH)
    y_pred_tfidf = tfidf_model.predict(reviews)

    acc_tfidf = accuracy_score(y_true, y_pred_tfidf)
    prec_tfidf = precision_score(y_true, y_pred_tfidf, pos_label="positive")
    rec_tfidf = recall_score(y_true, y_pred_tfidf, pos_label="positive")
    f1_tfidf = f1_score(y_true, y_pred_tfidf, pos_label="positive")

    # 2. Evaluate DistilBERT Transformer
    print("2. Evaluating DistilBERT model...")
    nlp = pipeline(
        "text-classification",
        model=TRANSFORMER_MODEL_DIR,
        tokenizer=TRANSFORMER_MODEL_DIR,
        truncation=True,
        max_length=256,
        batch_size=32
    )

    transformer_outputs = nlp(reviews)
    y_pred_trans = [out["label"].lower() for out in transformer_outputs]

    acc_trans = accuracy_score(y_true, y_pred_trans)
    prec_trans = precision_score(y_true, y_pred_trans, pos_label="positive")
    rec_trans = recall_score(y_true, y_pred_trans, pos_label="positive")
    f1_trans = f1_score(y_true, y_pred_trans, pos_label="positive")

    # Display comparison table
    results_df = pd.DataFrame([
        {
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": f"{acc_tfidf:.4f}",
            "Precision": f"{prec_tfidf:.4f}",
            "Recall": f"{rec_tfidf:.4f}",
            "F1-Score": f"{f1_tfidf:.4f}"
        },
        {
            "Model": "DistilBERT (Fine-tuned)",
            "Accuracy": f"{acc_trans:.4f}",
            "Precision": f"{prec_trans:.4f}",
            "Recall": f"{rec_trans:.4f}",
            "F1-Score": f"{f1_trans:.4f}"
        }
    ])

    print("\n" + "=" * 65)
    print("            MODEL COMPARISON ON 1,000 TEST REVIEWS")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
