import os
import re
import joblib
import streamlit as st

TFIDF_MODEL_PATH = "tfidf_model.joblib"
TRANSFORMER_MODEL_DIR = "transformer_model"


# Text cleaning function: remove HTML tags and lowercase
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# Cache models to load them once into memory
@st.cache_resource
def load_tfidf_model():
    if os.path.exists(TFIDF_MODEL_PATH):
        return joblib.load(TFIDF_MODEL_PATH)
    return None


@st.cache_resource
def load_transformer_pipeline():
    # Only load DistilBERT if the model folder exists
    if os.path.exists(TRANSFORMER_MODEL_DIR):
        try:
            import torch
            from transformers import pipeline
            return pipeline(
                "text-classification",
                model=TRANSFORMER_MODEL_DIR,
                tokenizer=TRANSFORMER_MODEL_DIR,
                truncation=True,
                max_length=256,
            )
        except ImportError:
            return None
    return None


def main():
    st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")

    # Custom CSS for dark gradient background, centered layout, and styled widgets
    st.markdown(
        """
        <style>
        /* Gradient background from dark navy to deep indigo */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            background-attachment: fixed;
        }

        /* Centered content container with max-width around 760px */
        .block-container {
            max-width: 760px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            margin: 0 auto;
        }

        /* Centered title styling */
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }

        /* Subtitle under title */
        .sub-title {
            text-align: center;
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 1.75rem;
        }

        /* Rounded text area with purple focus glow */
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #334155;
            background-color: #1e293b;
            color: #f1f5f9;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .stTextArea textarea:focus {
            border-color: #8b5cf6;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.35);
            outline: none;
        }

        /* Rounded buttons with hover effect */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        }

        /* Purple gradient on the main Predict button (primary) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
            border: none;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #9d74f7 0%, #7c3aed 100%);
            box-shadow: 0 6px 18px rgba(139, 92, 246, 0.5);
        }

        /* Hide Streamlit footer and main menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Styled prediction card */
        .result-card {
            border-radius: 14px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }
        .result-card-positive {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .result-card-negative {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        .result-label-positive {
            font-size: 1.8rem;
            font-weight: 800;
            color: #34d399;
            margin-bottom: 0.3rem;
        }
        .result-label-negative {
            font-size: 1.8rem;
            font-weight: 800;
            color: #f87171;
            margin-bottom: 0.3rem;
        }
        .result-confidence {
            font-size: 1.15rem;
            font-weight: 600;
            color: #e2e8f0;
        }

        /* Footer line */
        .app-footer {
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;
            margin-top: 3.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and subtitle
    st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter a movie review below to predict whether the sentiment is Positive or Negative.</div>',
        unsafe_allow_html=True,
    )

    # Model selection (check if DistilBERT model folder exists)
    has_transformer = os.path.exists(TRANSFORMER_MODEL_DIR)

    if has_transformer:
        model_choice = st.radio(
            "Select Model Architecture:",
            ("TF-IDF + Logistic Regression", "DistilBERT (Transformer)"),
            horizontal=True,
        )
    else:
        model_choice = st.radio(
            "Select Model Architecture:",
            ("TF-IDF + Logistic Regression",),
            horizontal=True,
        )
        st.caption("The DistilBERT demo runs locally; see the GitHub README.")

    # Input text area
    user_review = st.text_area(
        "Movie Review Text:",
        height=150,
        placeholder="Paste your movie review here (e.g., 'This movie was absolutely fantastic, with stellar acting and a captivating plot!')...",
    )

    # Quick test examples
    st.write("Or try a sample review:")
    col1, col2 = st.columns(2)
    sample_pos = "An absolute masterpiece! The cinematography was breathtaking and the performances were phenomenal."
    sample_neg = "Terrible movie. The acting was bland, the pacing was agonizingly slow, and the script made no sense."

    if col1.button("Load Positive Example"):
        user_review = sample_pos
        st.session_state["review_input"] = sample_pos
    if col2.button("Load Negative Example"):
        user_review = sample_neg
        st.session_state["review_input"] = sample_neg

    if "review_input" in st.session_state and not user_review:
        user_review = st.session_state["review_input"]

    # Prediction button
    if st.button("Predict Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a movie review first.")
            return

        cleaned_input = clean_text(user_review)

        # Handle TF-IDF selection
        if model_choice == "TF-IDF + Logistic Regression":
            tfidf_model = load_tfidf_model()
            if tfidf_model is None:
                st.error(
                    f"Model file `{TFIDF_MODEL_PATH}` not found! Please train it first by running: `python train_tfidf.py`"
                )
                return

            # Get predicted probabilities
            probabilities = tfidf_model.predict_proba([cleaned_input])[0]
            classes = list(tfidf_model.classes_)
            best_idx = probabilities.argmax()
            predicted_label = classes[best_idx].lower()
            confidence = probabilities[best_idx] * 100

        # Handle Transformer selection
        else:
            trans_pipe = load_transformer_pipeline()
            if trans_pipe is None:
                st.error(
                    f"Transformer folder `{TRANSFORMER_MODEL_DIR}` not found! Please train it first by running: `python train_transformer.py`"
                )
                return

            result = trans_pipe(cleaned_input)[0]
            predicted_label = result["label"].lower()
            confidence = result["score"] * 100

        # Display results as a styled card
        if predicted_label == "positive":
            card_html = f"""
            <div class="result-card result-card-positive">
                <div class="result-label-positive">Positive Sentiment 👍</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """
        else:
            card_html = f"""
            <div class="result-card result-card-negative">
                <div class="result-label-negative">Negative Sentiment 👎</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """
        st.markdown(card_html, unsafe_allow_html=True)
        st.progress(min(max(confidence / 100.0, 0.0), 1.0))

    # Footer line
    st.markdown(
        '<div class="app-footer">Built with scikit-learn, Hugging Face Transformers and Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
