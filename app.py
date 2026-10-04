import os
import re
import html
import joblib
import pandas as pd
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


# Explain TF-IDF prediction by calculating feature contributions
def explain_tfidf_prediction(tfidf_pipeline, raw_review, cleaned_review):
    vectorizer = tfidf_pipeline.named_steps["tfidf"]
    classifier = tfidf_pipeline.named_steps["clf"]

    # Transform the single review using the fitted TF-IDF vectorizer
    x = vectorizer.transform([cleaned_review])
    cols = x.nonzero()[1]
    coefs = classifier.coef_[0]
    words = vectorizer.get_feature_names_out()

    contribs = {}
    word_scores = []
    for c in cols:
        w = words[c]
        val = float(x[0, c] * coefs[c])
        contribs[w] = val
        word_scores.append((w, val))

    # Top 5 impactful words by absolute contribution
    word_scores.sort(key=lambda item: abs(item[1]), reverse=True)
    top_5 = word_scores[:5]

    # Split the original user text while preserving all whitespace and punctuation
    tokens = re.split(r"(\b[a-zA-Z0-9_\'-]+\b)", raw_review)
    annotated_tokens = []
    for tok in tokens:
        escaped_tok = html.escape(tok)
        tok_lower = tok.lower()
        if tok_lower in contribs:
            score = contribs[tok_lower]
            if score > 0.05:
                annotated_tokens.append(
                    f'<span class="hl-token hl-pos" title="Positive impact: +{score:.2f}">{escaped_tok}</span>'
                )
            elif score < -0.05:
                annotated_tokens.append(
                    f'<span class="hl-token hl-neg" title="Negative impact: {score:.2f}">{escaped_tok}</span>'
                )
            else:
                annotated_tokens.append(escaped_tok)
        else:
            annotated_tokens.append(escaped_tok)

    highlighted_html = "".join(annotated_tokens)
    return highlighted_html, top_5


# Callback to update review text in session state
def set_sample_review(text):
    st.session_state["review_text"] = text


def main():
    st.set_page_config(
        page_title="ReelSense • Movie Review Sentiment Portal",
        page_icon="🎬",
        layout="centered",
    )

    # -------------------------------------------------------------
    # Unified CSS Styling Block (Fintech Portal Aesthetic)
    # -------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Base page background */
        .stApp {
            background: linear-gradient(180deg, #0b0f19 0%, #0f172a 50%, #090d16 100%);
            background-attachment: fixed;
            color: #f1f5f9;
        }

        /* Centered content constraint */
        .block-container {
            max-width: 820px;
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            margin: 0 auto;
        }

        /* Top Navbar */
        .navbar-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
        .navbar-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-tile {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35);
        }
        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        .brand-subtitle {
            font-size: 0.75rem;
            color: #94a3b8;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #34d399;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            display: inline-block;
        }

        /* Hero Section */
        .hero-section {
            text-align: center;
            margin-bottom: 2.25rem;
        }
        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(139, 92, 246, 0.12);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 9999px;
            padding: 5px 16px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #c4b5fd;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.5;
        }

        /* Main Card Container with top accent line */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #1e293b !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35) !important;
            position: relative !important;
            overflow: hidden !important;
            padding: 1.25rem 1.5rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
            z-index: 10;
        }

        /* Section Header with Icon Tile */
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .header-tile {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .header-tile-purple {
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        .header-tile-blue {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
        }
        .header-tile-green {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .header-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f8fafc;
        }

        /* Quick Presets chip buttons */
        div[data-testid="stHorizontalBlock"] .stButton > button {
            border-radius: 20px !important;
            font-size: 0.825rem !important;
            font-weight: 600 !important;
            padding: 0.35rem 0.75rem !important;
            background: #0f172a !important;
            border: 1px solid #334155 !important;
            color: #cbd5e1 !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            border-color: #8b5cf6 !important;
            color: #ffffff !important;
            background: #1e1b4b !important;
            transform: translateY(-1px);
        }

        /* Text Area */
        .stTextArea textarea {
            border-radius: 14px !important;
            border: 1px solid #334155 !important;
            background-color: #0f172a !important;
            color: #f8fafc !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stTextArea textarea:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.3) !important;
            outline: none !important;
        }

        /* Primary Action Button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
            border-radius: 12px !important;
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.65rem 1.5rem !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
            transition: all 0.2s ease-in-out !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #9d74f7 0%, #4f46e5 100%) !important;
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.55) !important;
            transform: translateY(-1px) !important;
        }

        /* Result Card */
        .result-card {
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        }
        .result-card-pos {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .result-card-neg {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.25) 100%);
            border: 1px solid rgba(239, 68, 68, 0.4);
        }
        .result-head-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 10px;
        }
        .result-title-pos {
            font-size: 1.75rem;
            font-weight: 800;
            color: #34d399;
        }
        .result-title-neg {
            font-size: 1.75rem;
            font-weight: 800;
            color: #f87171;
        }
        .result-badge {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 9999px;
            padding: 6px 16px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        /* Custom Dual Probability Progress Bars */
        .prob-wrapper {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .prob-row {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .prob-info {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .prob-name-pos { color: #34d399; }
        .prob-name-neg { color: #f87171; }
        .prob-num { color: #f8fafc; font-weight: 700; }
        .custom-bar-track {
            height: 10px;
            background: #0f172a;
            border-radius: 9999px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .custom-bar-fill-pos {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            border-radius: 9999px;
            transition: width 0.4s ease;
        }
        .custom-bar-fill-neg {
            height: 100%;
            background: linear-gradient(90deg, #ef4444, #f87171);
            border-radius: 9999px;
            transition: width 0.4s ease;
        }

        /* "Why this prediction?" Explanability Box */
        .explain-box {
            background: #0f172a;
            border-radius: 14px;
            border: 1px solid #334155;
            padding: 1.25rem;
            margin-top: 1.25rem;
        }
        .explain-heading {
            font-size: 0.95rem;
            font-weight: 700;
            color: #c4b5fd;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .annotated-text-box {
            font-size: 0.95rem;
            line-height: 1.7;
            color: #e2e8f0;
            background: #1e293b;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #334155;
            margin-bottom: 1rem;
        }
        .hl-token {
            padding: 2px 5px;
            border-radius: 4px;
            font-weight: 600;
        }
        .hl-pos {
            background: rgba(16, 185, 129, 0.25);
            color: #6ee7b7;
            border-bottom: 2px solid #10b981;
        }
        .hl-neg {
            background: rgba(239, 68, 68, 0.25);
            color: #fca5a5;
            border-bottom: 2px solid #ef4444;
        }
        .chips-container {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
        }
        .chips-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94a3b8;
        }
        .word-chip {
            font-size: 0.8rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 9999px;
        }
        .word-chip-pos {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #6ee7b7;
        }
        .word-chip-neg {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 41, 59, 0.5);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
            padding: 8px 16px;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background: #8b5cf6 !important;
            color: #ffffff !important;
        }

        /* Info Card in About & Compare */
        .portal-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        /* Hide Streamlit Header, Footer & Menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

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

    # -------------------------------------------------------------
    # 1. Top Navbar
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="navbar-container">
            <div class="navbar-left">
                <div class="logo-tile">🎬</div>
                <div>
                    <div class="brand-title">ReelSense</div>
                    <div class="brand-subtitle">Movie Review Sentiment Portal</div>
                </div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Production Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # 2. Hero Section
    # -------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-section">
            <div class="pill-badge">FREE INSTANT ANALYSIS • TF-IDF + DISTILBERT COMPARISON</div>
            <div class="hero-title">Evaluate Movie Sentiment in Real-Time</div>
            <div class="hero-subtitle">
                Compare TF-IDF and DistilBERT on 50,000 IMDB reviews, with word-level explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for review text before widgets
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""

    # -------------------------------------------------------------
    # 8. Navigation Tabs (Predict, Model Comparison, About)
    # -------------------------------------------------------------
    tab_predict, tab_compare, tab_about = st.tabs(["⚡ Predict", "📊 Model Comparison", "ℹ️ About"])

    with tab_predict:
        # 3. Main Card Container with Top Accent Line
        with st.container(border=True):
            # Model Architecture Section
            st.markdown(
                """
                <div class="section-header">
                    <div class="header-tile header-tile-purple">⚙️</div>
                    <div class="header-title">Model Architecture</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            has_transformer = os.path.exists(TRANSFORMER_MODEL_DIR)
            if has_transformer:
                model_choice = st.radio(
                    "Select Model Architecture:",
                    ("TF-IDF + Logistic Regression", "DistilBERT (Transformer)"),
                    horizontal=True,
                    label_visibility="collapsed",
                )
            else:
                model_choice = st.radio(
                    "Select Model Architecture:",
                    ("TF-IDF + Logistic Regression",),
                    horizontal=True,
                    label_visibility="collapsed",
                )
                st.caption("The DistilBERT demo runs locally; see the GitHub README.")

            # Your Review Section Header
            st.markdown(
                """
                <div class="section-header" style="margin-top: 1.25rem;">
                    <div class="header-tile header-tile-blue">📝</div>
                    <div class="header-title">Your Review</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 4. Quick Presets Row styled as Chips
            sample_pos = "An absolute masterpiece! The cinematography was breathtaking and the performances were phenomenal."
            sample_neg = "Terrible movie. The acting was bland, the pacing was agonizingly slow, and the script made no sense."
            sample_mixed = "The visuals and special effects were stunning, but the plot was predictable and the characters lacked depth."

            st.write('<span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">Quick Presets:</span>', unsafe_allow_html=True)
            chip1, chip2, chip3, chip4 = st.columns(4)
            chip1.button("✨ Positive", on_click=set_sample_review, args=(sample_pos,), use_container_width=True)
            chip2.button("⚠️ Negative", on_click=set_sample_review, args=(sample_neg,), use_container_width=True)
            chip3.button("⚖️ Mixed", on_click=set_sample_review, args=(sample_mixed,), use_container_width=True)
            chip4.button("🔄 Reset", on_click=set_sample_review, args=("",), use_container_width=True)

            # Text Area for Review
            user_review = st.text_area(
                "Movie Review Text:",
                key="review_text",
                height=150,
                placeholder="Paste or type a movie review here (e.g., 'An absolute cinematic masterpiece with outstanding performances!')...",
                label_visibility="collapsed",
            )

            # Predict Button
            predict_clicked = st.button("⚡ Run Sentiment Analysis", type="primary", use_container_width=True)

        # Handle Sentiment Prediction
        if predict_clicked:
            if not user_review.strip():
                st.warning("Please enter a movie review first.")
            else:
                cleaned_input = clean_text(user_review)

                # TF-IDF Pipeline Path
                if model_choice == "TF-IDF + Logistic Regression":
                    tfidf_model = load_tfidf_model()
                    if tfidf_model is None:
                        st.error(
                            f"Model file `{TFIDF_MODEL_PATH}` not found! Please train it first: `python train_tfidf.py`"
                        )
                    else:
                        probabilities = tfidf_model.predict_proba([cleaned_input])[0]
                        classes = list(tfidf_model.classes_)
                        pos_idx = classes.index("positive") if "positive" in classes else 1
                        neg_idx = classes.index("negative") if "negative" in classes else 0

                        pos_prob = probabilities[pos_idx] * 100
                        neg_prob = probabilities[neg_idx] * 100

                        best_idx = probabilities.argmax()
                        predicted_label = classes[best_idx].lower()
                        confidence = probabilities[best_idx] * 100

                        # 5. Section Header for Prediction
                        st.markdown(
                            """
                            <div class="section-header" style="margin-top: 1.5rem;">
                                <div class="header-tile header-tile-green">🎯</div>
                                <div class="header-title">Prediction</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 6. Result Card with large label, confidence and dual progress bars
                        card_class = "result-card-pos" if predicted_label == "positive" else "result-card-neg"
                        title_class = "result-title-pos" if predicted_label == "positive" else "result-title-neg"
                        sentiment_text = "Positive Sentiment 👍" if predicted_label == "positive" else "Negative Sentiment 👎"

                        st.markdown(
                            f"""
                            <div class="result-card {card_class}">
                                <div class="result-head-row">
                                    <div class="{title_class}">{sentiment_text}</div>
                                    <div class="result-badge">{confidence:.2f}% Confidence</div>
                                </div>
                                <div class="prob-wrapper">
                                    <div class="prob-row">
                                        <div class="prob-info">
                                            <span class="prob-name-pos">Positive Probability</span>
                                            <span class="prob-num">{pos_prob:.1f}%</span>
                                        </div>
                                        <div class="custom-bar-track">
                                            <div class="custom-bar-fill-pos" style="width: {pos_prob:.1f}%;"></div>
                                        </div>
                                    </div>
                                    <div class="prob-row">
                                        <div class="prob-info">
                                            <span class="prob-name-neg">Negative Probability</span>
                                            <span class="prob-num">{neg_prob:.1f}%</span>
                                        </div>
                                        <div class="custom-bar-track">
                                            <div class="custom-bar-fill-neg" style="width: {neg_prob:.1f}%;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 7. "Why this prediction?" Section (TF-IDF Explainability)
                        highlighted_html, top_5 = explain_tfidf_prediction(tfidf_model, user_review, cleaned_input)

                        chips_html_list = []
                        for word, score in top_5:
                            escaped_word = html.escape(word)
                            if score > 0:
                                chips_html_list.append(
                                    f'<span class="word-chip word-chip-pos">+{score:.2f} {escaped_word}</span>'
                                )
                            else:
                                chips_html_list.append(
                                    f'<span class="word-chip word-chip-neg">{score:.2f} {escaped_word}</span>'
                                )
                        chips_html = "".join(chips_html_list)

                        st.markdown(
                            f"""
                            <div class="explain-box">
                                <div class="explain-heading">💡 Why this prediction? (Feature Attribution)</div>
                                <div class="annotated-text-box">
                                    {highlighted_html}
                                </div>
                                <div class="chips-container">
                                    <span class="chips-label">Top Predictive Drivers:</span>
                                    {chips_html}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Transformer Pipeline Path
                else:
                    trans_pipe = load_transformer_pipeline()
                    if trans_pipe is None:
                        st.error(
                            f"Transformer folder `{TRANSFORMER_MODEL_DIR}` not found! Please train it first: `python train_transformer.py`"
                        )
                    else:
                        result = trans_pipe(cleaned_input)[0]
                        predicted_label = result["label"].lower()
                        confidence = result["score"] * 100

                        if predicted_label == "positive":
                            pos_prob = confidence
                            neg_prob = 100.0 - confidence
                        else:
                            neg_prob = confidence
                            pos_prob = 100.0 - confidence

                        # 5. Section Header for Prediction
                        st.markdown(
                            """
                            <div class="section-header" style="margin-top: 1.5rem;">
                                <div class="header-tile header-tile-green">🎯</div>
                                <div class="header-title">Prediction</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 6. Result Card
                        card_class = "result-card-pos" if predicted_label == "positive" else "result-card-neg"
                        title_class = "result-title-pos" if predicted_label == "positive" else "result-title-neg"
                        sentiment_text = "Positive Sentiment 👍" if predicted_label == "positive" else "Negative Sentiment 👎"

                        st.markdown(
                            f"""
                            <div class="result-card {card_class}">
                                <div class="result-head-row">
                                    <div class="{title_class}">{sentiment_text}</div>
                                    <div class="result-badge">{confidence:.2f}% Confidence</div>
                                </div>
                                <div class="prob-wrapper">
                                    <div class="prob-row">
                                        <div class="prob-info">
                                            <span class="prob-name-pos">Positive Probability</span>
                                            <span class="prob-num">{pos_prob:.1f}%</span>
                                        </div>
                                        <div class="custom-bar-track">
                                            <div class="custom-bar-fill-pos" style="width: {pos_prob:.1f}%;"></div>
                                        </div>
                                    </div>
                                    <div class="prob-row">
                                        <div class="prob-info">
                                            <span class="prob-name-neg">Negative Probability</span>
                                            <span class="prob-num">{neg_prob:.1f}%</span>
                                        </div>
                                        <div class="custom-bar-track">
                                            <div class="custom-bar-fill-neg" style="width: {neg_prob:.1f}%;"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # -------------------------------------------------------------
    # Tab 2: Model Comparison
    # -------------------------------------------------------------
    with tab_compare:
        st.markdown(
            """
            <div class="section-header">
                <div class="header-tile header-tile-purple">📊</div>
                <div class="header-title">Comparative Performance Benchmark</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        comparison_df = pd.DataFrame([
            {
                "Model": "TF-IDF + Logistic Regression",
                "Accuracy": 0.9070,
                "Precision": 0.8916,
                "Recall": 0.9160,
                "F1": 0.9036,
            },
            {
                "Model": "DistilBERT (fine-tuned)",
                "Accuracy": 0.8760,
                "Precision": 0.8877,
                "Recall": 0.8466,
                "F1": 0.8667,
            },
        ])

        # Display exact values in styled dataframe
        st.dataframe(
            comparison_df.set_index("Model").style.format("{:.4f}"),
            use_container_width=True,
        )

        # Bar chart comparing Accuracy and F1-Score
        st.markdown(
            """
            <div class="section-header" style="margin-top: 1rem;">
                <div class="header-tile header-tile-blue">📈</div>
                <div class="header-title">Accuracy & F1-Score Comparison</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        chart_data = comparison_df[["Model", "Accuracy", "F1"]].set_index("Model")
        st.bar_chart(chart_data)

        # Hardware and sample constraint note
        st.info(
            "ℹ️ **Note on DistilBERT Performance:** DistilBERT was fine-tuned on a laptop-friendly subset of only **4,000 training reviews** for **1 epoch on a CPU**. When scaled to all 40,000 training reviews across multiple epochs on GPU hardware, Transformer models typically achieve >93% accuracy."
        )

    # -------------------------------------------------------------
    # Tab 3: About
    # -------------------------------------------------------------
    with tab_about:
        st.markdown(
            """
            <div class="section-header">
                <div class="header-tile header-tile-purple">ℹ️</div>
                <div class="header-title">About ReelSense</div>
            </div>
            <div class="portal-card">
                <p><strong>ReelSense</strong> is a production-grade movie review sentiment intelligence portal built for binary sentiment classification on the Kaggle IMDB 50,000 Movie Reviews dataset.</p>
                <p>This platform compares two distinct paradigms in Natural Language Processing:</p>
                <ul>
                    <li><strong>Classical Machine Learning:</strong> TF-IDF n-gram vectorization with L2-regularized Logistic Regression, providing ultra-low inference latency (~2ms) and transparent word attribution.</li>
                    <li><strong>Transformer Deep Learning:</strong> Fine-tuned DistilBERT (Bidirectional Encoder Representations from Transformers) with self-attention for deep contextual nuance.</li>
                </ul>
                <p style="margin-top: 1.25rem;">
                    🔗 <strong>GitHub Repository:</strong> <a href="https://github.com/sowmiST/movie-sentiment-analysis" target="_blank" style="color: #a78bfa; text-decoration: underline;">https://github.com/sowmiST/movie-sentiment-analysis</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Footer Line
    # -------------------------------------------------------------
    st.markdown(
        '<div class="app-footer">Built with scikit-learn, Hugging Face Transformers and Streamlit</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
