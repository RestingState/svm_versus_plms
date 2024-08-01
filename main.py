import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import ipdb;

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._preprocess)

    def _preprocess(self, text: str):
        # Remove numbers and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Lowercasing
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        return ' '.join(tokens)
    
df = pd.read_csv('bbc-news-data.csv', sep='\t')
df = df.drop(['filename'], axis=1)

X = df['title'] + df['content']

category_categorical = pd.Categorical(df['category'])
y = category_categorical.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(TextPreprocessor(), TfidfVectorizer())

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

clf = LinearSVC().fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["A", "B", "C", "D", "F"]
)

score = accuracy_score(y_test, y_pred)

plt.title(f"Accuracy Score: {round(score, 3)}", size=15)
plt.savefig("./images/confusion-matix.png")