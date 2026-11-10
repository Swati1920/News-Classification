
# News Classification Using Machine Learning and NLP

## Project Overview

This project uses Natural Language Processing (NLP) and
Machine Learning algorithms to automatically classify
text documents into different news categories.

The system processes text, extracts important features
using TF-IDF, trains five ML algorithms and predicts
the category of user-provided text.

## Dataset

Dataset: 20 Newsgroups

Four selected categories:
- Computer Graphics
- Sports / Baseball
- Space / Science
- Politics

## NLP Techniques

- Text cleaning
- Lowercasing
- Tokenization through TF-IDF
- Stopword removal
- TF-IDF Vectorization
- Unigram and Bigram feature extraction

## Machine Learning Algorithms

1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Streamlit
- Joblib

## Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model is selected using
weighted F1-score.

## Project Execution

### 1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Install dependencies

pip install -r requirements.txt

### 3. Train the models

python main.py

### 4. Run the application

streamlit run app.py

## Output

The application accepts text input and predicts
the corresponding news category.


