import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    #  Lowercase
    text = str(text).lower()
    # Remove punctuation
    text = "".join([c for c in text if c not in string.punctuation])
    # Split into words
    tokens = text.split()
    # Remove stopwords
    cleaned_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)

def preprocess_data(df):
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    return X, vectorizer

def encode_labels(df):
    # Standardizing label names to 0 and 1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['label_num']
