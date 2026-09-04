"""
Clinical Trial Disease Category Classification - Streamlit App
-----------------------------------------------------------------
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------------------------------------------
# NLTK setup (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()


def clean_text(text: str) -> str:
    """Same cleaning pipeline used in the training notebook."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return " ".join(tokens)


# ----------------------------------------------------------------------
# Load trained model artifacts (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("disease_classifier_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    categories = joblib.load("disease_categories.pkl")
    return model, vectorizer, categories


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("clinical_trials_cleaned.csv")
    except FileNotFoundError:
        return None


try:
    model, vectorizer, categories = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

df = load_dataset()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🩺 Clinical Trial Disease Category Classifier")
st.markdown(
    "An NLP + Machine Learning system that predicts the **disease category** "
    "of a clinical trial from its **Brief Summary** text."
)

if not artifacts_loaded:
    st.error(
        "Model files not found. Please run all cells in the notebook "
        "`Clinical_Trial_Disease_Classification.ipynb` first — this generates "
        "`disease_classifier_model.pkl`, `tfidf_vectorizer.pkl`, and "
        "`disease_categories.pkl` in this same folder."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA Dashboard", "ℹ️ About"])

# ----------------------------------------------------------------------
# TAB 1 — Prediction
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Predict Disease Category from a Clinical Trial Summary")

    example = (
        "This study evaluates the effect of a new insulin regimen on glycemic "
        "control in adult patients with elevated blood sugar and insulin resistance."
    )

    user_text = st.text_area(
        "Paste a clinical trial brief summary below:",
        value="",
        height=180,
        placeholder=example,
    )

    predict_clicked = st.button("Predict Disease Category", type="primary")

    if predict_clicked:
        if not user_text.strip():
            st.warning("Please enter or load some clinical trial summary text first.")
        else:
            cleaned = clean_text(user_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

            st.success(f"**Predicted Disease Category:** {prediction}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                proba_df = pd.DataFrame({
                    "Category": model.classes_,
                    "Probability": proba
                }).sort_values("Probability", ascending=False)

                fig = px.bar(
                    proba_df, x="Probability", y="Category",
                    orientation="h", color="Probability",
                    color_continuous_scale="Blues",
                    title="Prediction Confidence by Category"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(
                    "This model type doesn't expose probability scores — "
                    "only the top predicted class is shown."
                )

            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned)

# ----------------------------------------------------------------------
# TAB 2 — EDA Dashboard
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")

    if df is None:
        st.warning(
            "`clinical_trials_cleaned.csv` not found in this folder. "
            "Run the notebook's Data Preprocessing section to generate it, "
            "then reload this app to see the EDA dashboard."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trials", f"{len(df):,}")
        col2.metric("Disease Categories", df['source_condition_query'].nunique())
        col3.metric("Avg. Summary Length (words)", f"{df['word_count'].mean():.0f}" if 'word_count' in df.columns else "—")

        st.markdown("#### Disease Category Distribution")
        cat_counts = df['source_condition_query'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig1 = px.bar(cat_counts, x='Category', y='Count', color='Category',
                       title="Number of Clinical Trials per Disease Category")
        st.plotly_chart(fig1, use_container_width=True)

        if 'word_count' in df.columns:
            st.markdown("#### Summary Length by Category")
            fig2 = px.box(df, x='source_condition_query', y='word_count', color='source_condition_query',
                           title="Word Count Distribution per Category")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Sample Records")
        st.dataframe(df[['source_condition_query', 'brief_summary']].sample(min(10, len(df))), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 3 — About
# ----------------------------------------------------------------------
with tab3:
    st.subheader("About This Project")
    st.markdown(f"""
**Project:** Clinical Trial Disease Category Classification Using NLP and Machine Learning

**Pipeline:**
1. Data Collection — clinical trial dataset with `brief_summary` and `source_condition_query`
2. Data Preprocessing — cleaning, tokenization, stop-word removal, lemmatization
3. Exploratory Data Analysis — category distribution, word frequency, word clouds
4. NLP-Based Classification — TF-IDF vectorization + comparison of ML models
5. Disease Prediction System — best model saved and served here
6. This Streamlit app — interactive prediction + EDA dashboard

**Disease categories the model was trained on:**
{", ".join(categories)}

**Model in use:** `{type(model).__name__}`
""")
