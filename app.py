
import streamlit as st
import joblib
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

    with open(MODEL_DIR / "metadata.json", "r") as file:
        metadata = json.load(file)

    return model, vectorizer, metadata


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("📰 News Classification System")
st.caption("An intelligent news topic prediction tool powered by Machine Learning and NLP.")

st.divider()

if not (MODEL_DIR / "best_model.joblib").exists():
    st.error("Trained model not found. Please run main.py first.")
    st.stop()

model, vectorizer, metadata = load_model()

st.markdown("### Enter News Article")
st.write("Paste your news article below to identify its topic.")

user_text = st.text_area(
    "News Article",
    height=230,
    placeholder="Paste your complete news article here..."
)

if st.button("🔍 Classify News", type="primary", use_container_width=True):

    if len(user_text.strip()) < 20:
        st.warning("Please enter a meaningful news article (at least 20 characters).")

    else:
        cleaned_text = clean_text(user_text)

        if not cleaned_text:
            st.warning("The entered text does not contain enough valid text to classify.")

        else:
            text_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vector)[0]

            st.divider()
            st.markdown("### Classification Result")

            category_names = {
                "comp.graphics": "Computer Graphics",
                "rec.sport.baseball": "Sports – Baseball",
                "sci.space": "Space & Science",
                "talk.politics.misc": "Politics"
            }

            display_category = category_names.get(prediction, prediction)

            st.success("Article successfully classified!")

            st.markdown(
                f"""
                <div style="
                    background-color: #e8f3ff;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    border: 1px solid #b6d7f5;
                ">
                    <p style="color:#34516b; font-size:16px; margin-bottom:8px;">
                        PREDICTED NEWS CATEGORY
                    </p>
                    <h2 style="color:#174f82; margin:0;">
                        {display_category}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(f"Classification performed using {metadata['best_model']}.")

st.divider()

st.caption("News Classification using Machine Learning & Natural Language Processing")
