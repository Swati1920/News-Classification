
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


# 1. Create project directories

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in [DATASET_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# 2. Load dataset

print("\nLoading 20 Newsgroups dataset...")

categories = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc"
]

data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=42
)

df = pd.DataFrame({
    "text": data.data,
    "category": [
        data.target_names[i] for i in data.target
    ]
})

df.dropna(inplace=True)
df = df[df["text"].str.strip() != ""]
df.reset_index(drop=True, inplace=True)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nCategory distribution:")
print(df["category"].value_counts())


# Save dataset copy for project reference

df.to_csv(DATASET_DIR / "news_dataset.csv", index=False)


# 3. NLP preprocessing

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["text"].apply(clean_text)

print("\nNLP preprocessing completed!")


# 4. Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)


# 5. TF-IDF vectorization

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# 6. Define five ML algorithms

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": LinearSVC(
        random_state=42
    )
}


# 7. Train and evaluate models

results = []
trained_models = {}

print("\nTraining ML algorithms...\n")

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", round(accuracy * 100, 2), "%\n")


# 8. Compare model performance

results_df = pd.DataFrame(results)

results_df.sort_values(
    by="F1 Score",
    ascending=False,
    inplace=True
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))


# 9. Select best model

best_model_name = results_df.iloc[0]["Algorithm"]

best_model = trained_models[best_model_name]

print("\nBest performing model:", best_model_name)


# 10. Save model and vectorizer

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

metadata = {
    "best_model": best_model_name,
    "categories": list(data.target_names),
    "total_documents": len(df),
    "training_documents": len(X_train),
    "testing_documents": len(X_test)
}

with open(MODEL_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)


# 11. Generate performance comparison graph

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.title("ML Algorithm Accuracy Comparison")

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.ylim(0, 1)
plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png"
)

plt.close()


# 12. Generate confusion matrix

best_predictions = best_model.predict(X_test_tfidf)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Best Model Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.close()


# 13. Classification report

report = classification_report(
    y_test,
    best_predictions,
    zero_division=0
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)


print("\nPROJECT TRAINING COMPLETED!")
print("Best model:", best_model_name)
print("\nAll models evaluated successfully.")
print("Models saved in models/ folder.")
print("Results saved in outputs/ folder.")
