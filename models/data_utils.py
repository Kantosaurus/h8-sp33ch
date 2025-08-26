import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import string

def load_and_preprocess_data(data_path='../data/combined.csv', test_size=0.2, random_state=42):
    """
    Load and preprocess the hate speech dataset
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic text preprocessing
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        # Convert to lowercase
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Preprocess text
    df['processed_text'] = df['post'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]
    
    # Features and labels
    X_text = df['processed_text'].values
    y = df['label'].values
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(X_text).toarray()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Positive class ratio: {y.mean():.3f}")
    
    return X_train, X_test, y_train, y_test, vectorizer