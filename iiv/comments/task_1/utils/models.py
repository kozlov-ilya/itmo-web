import random
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


# nltk.download('punkt_tab')
# nltk.download('stopwords')

models = [
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "CountVectorizer",
            "ngrams": "1",
        },
        "model": LogisticRegression(random_state=random.randint(1, 100000), max_iter=1000),
        "vectorizer": CountVectorizer(ngram_range=(1, 1))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "CountVectorizer",
            "ngrams": "3",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": CountVectorizer(ngram_range=(3, 3))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "TF-IDF",
            "ngrams": "1",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": TfidfVectorizer(ngram_range=(1, 1))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "TF-IDF",
            "ngrams": "2",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": TfidfVectorizer(ngram_range=(2, 2))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "TF-IDF",
            "ngrams": "3",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": TfidfVectorizer(ngram_range=(3, 3))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "TF-IDF",
            "ngrams": "5",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": TfidfVectorizer(ngram_range=(5, 5))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "CountVectorizer",
            "ngrams": "1 - 2",
            "tokenizer": "word_tokenize",
            "noise": "stop_w + punct"
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": CountVectorizer(ngram_range=(1, 2), tokenizer=word_tokenize, stop_words=stopwords.words('russian') + list(punctuation))
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "TF-IDF",
            "ngrams": "1",
            "tokenizer": "word_tokenize",
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": TfidfVectorizer(ngram_range=(1, 1), tokenizer=word_tokenize)
    },
    {
        "labels": {
            "model": 'LogisticRegression',
            "vectorizer": "CountVectorizer",
            "ngrams": "1",
            "analyzer": "char"
        },
        "model": LogisticRegression(random_state=random.randint(1, 10000), max_iter=1000),
        "vectorizer": CountVectorizer(ngram_range=(1, 1), analyzer='char')
    },
    {
        "labels": {
            "model": 'RandomForestClassifier',
            "vectorizer": "CountVectorizer",
            "ngrams": "1",
        },
        "model": RandomForestClassifier(n_estimators=50, max_depth=50, min_samples_split=5, random_state=42),
        "vectorizer": CountVectorizer(ngram_range=(1, 1))
    },
    {
        "labels": {
            "model": 'KNeighborsClassifier',
            "vectorizer": "CountVectorizer",
            "ngrams": "1",
        },
        "model": KNeighborsClassifier(n_neighbors=2),
        "vectorizer": CountVectorizer(ngram_range=(1, 1))
    },
    {
        "labels": {
            "model": 'DecisionTreeClassifier',
            "vectorizer": "CountVectorizer",
            "ngrams": "1",
        },
        "model": DecisionTreeClassifier(max_depth=50, min_samples_split=10, random_state=42),
        "vectorizer": CountVectorizer(ngram_range=(1, 1))
    }
]
