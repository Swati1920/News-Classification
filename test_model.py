
import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "best_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

test_articles = {
    "COMPUTER GRAPHICS": """
    Computer graphics researchers are developing advanced 3D rendering
    techniques for digital animation. Graphics software uses pixels,
    image processing, visual effects and computer-generated models
    to create realistic images on computer screens.
    """,

    "SPORTS BASEBALL": """
    The baseball team won the championship after scoring a home run
    in the final inning. The pitcher struck out three batters while
    the players celebrated their victory at the baseball stadium.
    """,

    "SPACE AND SCIENCE": """
    NASA launched a spacecraft to study distant planets and explore
    the solar system. Scientists are examining satellite images,
    space missions, planetary orbits and the possibility of life
    beyond Earth.
    """,

    "POLITICS": """
    Members of parliament debated a new government bill before the
    upcoming election. The president and political leaders discussed
    public policy, legislation, voting rights and national governance.
    """
}

print("\nMODEL PREDICTION TEST")
print("-" * 50)

for actual_topic, article in test_articles.items():
    cleaned = clean_text(article)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    print("\nTest Article:", actual_topic)
    print("Predicted Category:", prediction)

print("\nModel classes:", model.classes_)
