import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -------------------------------------------------------------
# Configuration Variables (subset sizes for laptop training)
# -------------------------------------------------------------
TRAIN_SUBSET_SIZE = 4000
TEST_SUBSET_SIZE = 1000
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
NUM_EPOCHS = 1
BATCH_SIZE = 16
OUTPUT_DIR = "transformer_model"


# Text cleaning function: remove HTML tags and lowercase
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# Evaluation metrics calculation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    print("Loading IMDB Dataset...")
    df = pd.read_csv("IMDB Dataset.csv")
    print(f"Loaded {len(df)} reviews.")

    # Clean the review text
    print("Cleaning review text...")
    df["review"] = df["review"].apply(clean_text)

    # Map labels: negative -> 0, positive -> 1
    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}
    df["label"] = df["sentiment"].map(label2id)

    # 80/20 train/test split with random_state=42 (identical to TF-IDF split)
    print("Splitting dataset (80% train, 20% test)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Select subsets for training and testing
    train_subset = train_df.iloc[:TRAIN_SUBSET_SIZE]
    test_subset = test_df.iloc[:TEST_SUBSET_SIZE]
    print(f"Train subset size: {len(train_subset)}, Test subset size: {len(test_subset)}")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_subset[["review", "label"]])
    test_dataset = Dataset.from_pandas(test_subset[["review", "label"]])

    # Tokenize dataset
    print(f"Tokenizing reviews (max_length={MAX_LENGTH})...")
    def tokenize_function(examples):
        return tokenizer(examples["review"], truncation=True, max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load pre-trained DistilBERT model for sequence classification
    print(f"Loading pre-trained model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        seed=42,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    print("Fine-tuning DistilBERT (1 epoch)...")
    trainer.train()

    # Evaluate on the test subset
    print("\nEvaluating model on test subset...")
    eval_results = trainer.evaluate()
    test_acc = eval_results.get("eval_accuracy", 0.0)
    print(f"Test Accuracy: {test_acc:.4f}\n")

    # Save fine-tuned model and tokenizer
    print(f"Saving model and tokenizer to '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved successfully!")


if __name__ == "__main__":
    main()
