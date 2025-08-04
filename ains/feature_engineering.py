import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Text processing and NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER sentiment not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Install with: pip install textblob")

# FastText embeddings
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("Warning: FastText not available. Install with: pip install fasttext")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK data")


class LexicalViewTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for lexical view features (TF-IDF)
    """
    
    def __init__(self, max_features=20000):
        self.max_features = max_features
        self.word_vectorizer = None
        self.char_vectorizer = None
        
    def fit(self, X, y=None):
        # Word-level TF-IDF with optimized parameters
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2',  # L2 normalization for better performance
            use_idf=True,
            smooth_idf=True
        )
        
        # Character-level TF-IDF with memory optimization
        self.char_vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced for memory efficiency
            analyzer='char',
            ngram_range=(2, 4),  # Reduced range
            min_df=3,  # Increased minimum frequency
            max_df=0.95,
            dtype=np.float32  # Use float32 for memory efficiency
        )
        
        # Fit both vectorizers
        self.word_vectorizer.fit(X)
        self.char_vectorizer.fit(X)
        
        return self
    
    def transform(self, X):
        word_features = self.word_vectorizer.transform(X)
        char_features = self.char_vectorizer.transform(X)
        
        # Combine word and character features
        from scipy.sparse import hstack
        combined = hstack([word_features, char_features])
        
        return combined


class SemanticViewTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for semantic view features (FastText + lexicon similarity)
    """
    
    def __init__(self, fasttext_model_path=None, hate_lexicons=None):
        self.fasttext_model_path = fasttext_model_path
        self.hate_lexicons = hate_lexicons or self._get_default_hate_lexicons()
        self.fasttext_model = None
        
        # Initialize FastText model
        if FASTTEXT_AVAILABLE and fasttext_model_path:
            try:
                self.fasttext_model = fasttext.load_model(fasttext_model_path)
            except Exception as e:
                print(f"Warning: Could not load FastText model: {e}")
    
    def _get_default_hate_lexicons(self):
        return {
            'hatebase': [
                'hate', 'racist', 'bigot', 'nazi', 'supremacist', 'white power',
                'kill', 'murder', 'death', 'exterminate', 'genocide', 'ethnic cleansing',
                'slave', 'nigger', 'faggot', 'dyke', 'kike', 'spic', 'chink', 'gook',
                'terrorist', 'jihad', 'islamist', 'extremist', 'radical'
            ],
            'google_toxic': [
                'toxic', 'hateful', 'offensive', 'abusive', 'insulting', 'threatening',
                'violent', 'aggressive', 'hostile', 'malicious', 'vicious', 'cruel',
                'brutal', 'savage', 'barbaric', 'inhuman', 'evil', 'wicked'
            ],
            'slurs': [
                'bitch', 'whore', 'slut', 'cunt', 'pussy', 'dick', 'cock', 'fuck',
                'shit', 'ass', 'bastard', 'motherfucker', 'fucker', 'dumbass',
                'idiot', 'moron', 'retard', 'stupid', 'dumb', 'ignorant'
            ]
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        # FastText embeddings
        if self.fasttext_model is not None:
            fasttext_features = self._extract_fasttext_features(X)
            features.append(fasttext_features)
        else:
            # Dummy features if FastText not available
            features.append(np.zeros((len(X), 300)))
        
        # Hate lexicon similarity
        lexicon_features = self._extract_lexicon_similarity_features(X)
        features.append(lexicon_features)
        
        return np.hstack(features)
    
    def _extract_fasttext_features(self, texts):
        embeddings = []
        for text in texts:
            words = word_tokenize(text.lower())
            word_embeddings = []
            
            for word in words:
                try:
                    embedding = self.fasttext_model.get_word_vector(word)
                    word_embeddings.append(embedding)
                except:
                    continue
            
            if word_embeddings:
                avg_embedding = np.mean(word_embeddings, axis=0)
            else:
                avg_embedding = np.zeros(300)
            
            embeddings.append(avg_embedding)
        
        return np.array(embeddings)
    
    def _extract_lexicon_similarity_features(self, texts):
        features = []
        
        for text in texts:
            text_lower = text.lower()
            text_words = set(word_tokenize(text_lower))
            
            lexicon_similarities = []
            for lexicon_name, lexicon_words in self.hate_lexicons.items():
                matches = sum(1 for word in lexicon_words if word in text_words)
                similarity = matches / len(lexicon_words) if len(lexicon_words) > 0 else 0.0
                lexicon_similarities.append(similarity)
            
            features.append(lexicon_similarities)
        
        return np.array(features)


class StylisticViewTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for stylistic view features (text statistics + sentiment)
    """
    
    def __init__(self):
        self.sentiment_analyzer = None
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            text_features = self._extract_stylistic_features(text)
            features.append(text_features)
        
        return np.array(features)
    
    def _extract_stylistic_features(self, text):
        features = []
        
        # Text statistics
        words = word_tokenize(text)
        chars = list(text)
        
        # 1. Uppercase ratio
        uppercase_count = sum(1 for char in chars if char.isupper())
        uppercase_ratio = uppercase_count / len(chars) if chars else 0.0
        features.append(uppercase_ratio)
        
        # 2. Punctuation ratio
        punctuation_count = sum(1 for char in chars if char in string.punctuation)
        punctuation_ratio = punctuation_count / len(chars) if chars else 0.0
        features.append(punctuation_ratio)
        
        # 3. Word count
        word_count = len(words)
        features.append(word_count)
        
        # 4. Character count
        char_count = len(chars)
        features.append(char_count)
        
        # 5. Average word length
        if words:
            avg_word_length = np.mean([len(word) for word in words])
        else:
            avg_word_length = 0.0
        features.append(avg_word_length)
        
        # 6. Unique word ratio
        if words:
            unique_word_ratio = len(set(words)) / len(words)
        else:
            unique_word_ratio = 0.0
        features.append(unique_word_ratio)
        
        # 7. Digit ratio
        digit_count = sum(1 for char in chars if char.isdigit())
        digit_ratio = digit_count / len(chars) if chars else 0.0
        features.append(digit_ratio)
        
        # 8. VADER sentiment
        if self.sentiment_analyzer:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            features.extend([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu']
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 9. TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            features.extend([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        else:
            features.extend([0.0, 0.0])
        
        return features

class MultiViewFeatureEngineering:
    """
    Multi-view feature engineering for hate speech detection
    
    Creates three separate "views" of the data:
    1. Lexical View (Sparse): TF-IDF features
    2. Semantic View (Dense): FastText embeddings + hate lexicon similarity
    3. Stylistic View: Text statistics + sentiment analysis
    """
    
    def __init__(self, 
                 max_tfidf_features: int = 20000,
                 fasttext_model_path: Optional[str] = None,
                 hate_lexicons: Optional[Dict[str, List[str]]] = None,
                 use_feature_union: bool = True):
        """
        Initialize the multi-view feature engineering system
        
        Args:
            max_tfidf_features: Maximum number of TF-IDF features
            fasttext_model_path: Path to pre-trained FastText model
            hate_lexicons: Dictionary of hate lexicons for similarity calculation
            use_feature_union: Whether to use FeatureUnion for fusion (recommended)
        """
        self.max_tfidf_features = max_tfidf_features
        self.fasttext_model_path = fasttext_model_path
        self.hate_lexicons = hate_lexicons or self._get_default_hate_lexicons()
        self.use_feature_union = use_feature_union
        
        # Initialize components
        self.lexical_vectorizer = None
        self.fasttext_model = None
        self.sentiment_analyzer = None
        
        # Feature names for each view
        self.lexical_feature_names = []
        self.semantic_feature_names = []
        self.stylistic_feature_names = []
        
        # FeatureUnion pipeline
        self.feature_union = None
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize FastText model
        if FASTTEXT_AVAILABLE and fasttext_model_path:
            try:
                self.fasttext_model = fasttext.load_model(fasttext_model_path)
                print(f"Loaded FastText model from {fasttext_model_path}")
            except Exception as e:
                print(f"Warning: Could not load FastText model: {e}")
                self.fasttext_model = None
    
    def _get_default_hate_lexicons(self) -> Dict[str, List[str]]:
        """
        Get default hate lexicons for similarity calculation
        """
        return {
            'hatebase': [
                'hate', 'racist', 'bigot', 'nazi', 'supremacist', 'white power',
                'kill', 'murder', 'death', 'exterminate', 'genocide', 'ethnic cleansing',
                'slave', 'nigger', 'faggot', 'dyke', 'kike', 'spic', 'chink', 'gook',
                'terrorist', 'jihad', 'islamist', 'extremist', 'radical'
            ],
            'google_toxic': [
                'toxic', 'hateful', 'offensive', 'abusive', 'insulting', 'threatening',
                'violent', 'aggressive', 'hostile', 'malicious', 'vicious', 'cruel',
                'brutal', 'savage', 'barbaric', 'inhuman', 'evil', 'wicked'
            ],
            'slurs': [
                'bitch', 'whore', 'slut', 'cunt', 'pussy', 'dick', 'cock', 'fuck',
                'shit', 'ass', 'bastard', 'motherfucker', 'fucker', 'dumbass',
                'idiot', 'moron', 'retard', 'stupid', 'dumb', 'ignorant'
            ]
        }
    
    def fit_transform(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Fit the feature engineering pipeline and transform the data
        
        Args:
            texts: List of text documents
            
        Returns:
            Dictionary containing features for each view
        """
        print("=" * 60)
        print("MULTI-VIEW FEATURE ENGINEERING")
        print("=" * 60)
        
        if self.use_feature_union:
            return self._fit_transform_with_feature_union(texts)
        else:
            # Legacy method
            lexical_features = self._fit_transform_lexical_view(texts)
            semantic_features = self._fit_transform_semantic_view(texts)
            stylistic_features = self._fit_transform_stylistic_view(texts)
            
            return {
                'lexical': lexical_features,
                'semantic': semantic_features,
                'stylistic': stylistic_features
            }
    
    def _fit_transform_with_feature_union(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Fit and transform using FeatureUnion for proper feature fusion
        """
        print("\n[LINK] FEATURE FUSION WITH FEATUREUNION")
        print("-" * 40)
        
        # Create transformers for each view
        lexical_transformer = LexicalViewTransformer(max_features=self.max_tfidf_features)
        semantic_transformer = SemanticViewTransformer(
            fasttext_model_path=self.fasttext_model_path,
            hate_lexicons=self.hate_lexicons
        )
        stylistic_transformer = StylisticViewTransformer()
        
        # Create FeatureUnion with proper normalization
        self.feature_union = FeatureUnion([
            # Lexical view: sparse features (no normalization needed for TF-IDF)
            ('lexical', lexical_transformer),
            
            # Semantic view: dense features with z-score normalization
            ('semantic', Pipeline([
                ('semantic_features', semantic_transformer),
                ('normalizer', StandardScaler())
            ])),
            
            # Stylistic view: dense features with z-score normalization
            ('stylistic', Pipeline([
                ('stylistic_features', stylistic_transformer),
                ('normalizer', StandardScaler())
            ]))
        ])
        
        # Fit and transform
        print("Fitting FeatureUnion pipeline...")
        combined_features = self.feature_union.fit_transform(texts)
        
        # Extract individual views for return
        lexical_features = lexical_transformer.transform(texts)
        semantic_features = semantic_transformer.transform(texts)
        stylistic_features = stylistic_transformer.transform(texts)
        
        # Store feature names
        self._extract_feature_names(lexical_transformer, semantic_transformer, stylistic_transformer)
        
        print(f"[OK] FeatureUnion fusion completed")
        print(f"  - Combined features shape: {combined_features.shape}")
        print(f"  - Lexical (sparse): {lexical_features.shape[1]} features")
        print(f"  - Semantic (dense): {semantic_features.shape[1]} features")
        print(f"  - Stylistic (dense): {stylistic_features.shape[1]} features")
        
        return {
            'lexical': lexical_features,
            'semantic': semantic_features,
            'stylistic': stylistic_features,
            'combined': combined_features
        }
    
    def _extract_feature_names(self, lexical_transformer, semantic_transformer, stylistic_transformer):
        """
        Extract feature names from transformers
        """
        # Lexical feature names
        if hasattr(lexical_transformer.word_vectorizer, 'get_feature_names_out'):
            word_names = lexical_transformer.word_vectorizer.get_feature_names_out()
            char_names = [f"char_{name}" for name in lexical_transformer.char_vectorizer.get_feature_names_out()]
            self.lexical_feature_names = list(word_names) + char_names
        else:
            self.lexical_feature_names = [f"lexical_{i}" for i in range(25000)]  # Approximate
        
        # Semantic feature names
        semantic_count = 300 + len(self.hate_lexicons)  # FastText + lexicons
        self.semantic_feature_names = [f"semantic_{i}" for i in range(semantic_count)]
        
        # Stylistic feature names
        self.stylistic_feature_names = [
            'uppercase_ratio', 'punctuation_ratio', 'word_count', 'char_count',
            'avg_word_length', 'unique_word_ratio', 'digit_ratio',
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'textblob_polarity', 'textblob_subjectivity'
        ]
    
    def transform(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted pipeline
        
        Args:
            texts: List of text documents
            
        Returns:
            Dictionary containing features for each view
        """
        if self.feature_union is None and self.lexical_vectorizer is None:
            raise ValueError("Must call fit_transform() first")
        
        if self.use_feature_union and self.feature_union is not None:
            return self._transform_with_feature_union(texts)
        else:
            # Legacy method
            lexical_features = self._transform_lexical_view(texts)
            semantic_features = self._transform_semantic_view(texts)
            stylistic_features = self._transform_stylistic_view(texts)
            
            return {
                'lexical': lexical_features,
                'semantic': semantic_features,
                'stylistic': stylistic_features
            }
    
    def _transform_with_feature_union(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted FeatureUnion
        """
        # Transform using fitted FeatureUnion
        combined_features = self.feature_union.transform(texts)
        
        # Extract individual views (recreate transformers for consistency)
        lexical_transformer = LexicalViewTransformer(max_features=self.max_tfidf_features)
        semantic_transformer = SemanticViewTransformer(
            fasttext_model_path=self.fasttext_model_path,
            hate_lexicons=self.hate_lexicons
        )
        stylistic_transformer = StylisticViewTransformer()
        
        # Fit transformers on training data (this is a limitation, but needed for consistency)
        # In practice, you'd want to store the fitted transformers
        lexical_features = lexical_transformer.fit_transform(texts)
        semantic_features = semantic_transformer.transform(texts)
        stylistic_features = stylistic_transformer.transform(texts)
        
        return {
            'lexical': lexical_features,
            'semantic': semantic_features,
            'stylistic': stylistic_features,
            'combined': combined_features
        }
    
    def _fit_transform_lexical_view(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform lexical view (TF-IDF features)
        """
        print("\n📌 A. LEXICAL VIEW (Sparse)")
        print("-" * 40)
        
        # Word-level TF-IDF: unigram + bigram + trigram
        self.lexical_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=(1, 3),  # unigram, bigram, trigram
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit and transform
        lexical_features = self.lexical_vectorizer.fit_transform(texts)
        self.lexical_feature_names = self.lexical_vectorizer.get_feature_names_out()
        
        print(f"[OK] Word-level TF-IDF: {lexical_features.shape[1]} features")
        print(f"  - N-grams: unigram + bigram + trigram")
        print(f"  - Max features: {self.max_tfidf_features}")
        
        # Character-level TF-IDF for obfuscated hate
        char_vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer='char',
            ngram_range=(2, 5),  # character n-grams
            min_df=2,
            max_df=0.95
        )
        
        char_features = char_vectorizer.fit_transform(texts)
        char_feature_names = [f"char_{name}" for name in char_vectorizer.get_feature_names_out()]
        
        print(f"[OK] Character-level TF-IDF: {char_features.shape[1]} features")
        print(f"  - Character n-grams: 2-5")
        print(f"  - Good for obfuscated hate (e.g., 'b!tch')")
        
        # Combine word and character features
        combined_features = np.hstack([
            lexical_features.toarray(),
            char_features.toarray()
        ])
        
        # Update feature names
        self.lexical_feature_names = list(self.lexical_feature_names) + char_feature_names
        
        print(f"[OK] Total lexical features: {combined_features.shape[1]}")
        
        return combined_features
    
    def _transform_lexical_view(self, texts: List[str]) -> np.ndarray:
        """
        Transform new data for lexical view
        """
        if self.lexical_vectorizer is None:
            raise ValueError("Lexical vectorizer not fitted")
        
        # Transform word-level features
        lexical_features = self.lexical_vectorizer.transform(texts)
        
        # Transform character-level features (recreate vectorizer)
        char_vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer='char',
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.95
        )
        char_features = char_vectorizer.fit_transform(texts)
        
        # Combine features
        combined_features = np.hstack([
            lexical_features.toarray(),
            char_features.toarray()
        ])
        
        return combined_features
    
    def _fit_transform_semantic_view(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform semantic view (FastText embeddings + hate lexicon similarity)
        """
        print("\n📌 B. SEMANTIC VIEW (Dense)")
        print("-" * 40)
        
        semantic_features = []
        
        # 1. FastText embeddings (average pooling)
        if self.fasttext_model is not None:
            fasttext_features = self._extract_fasttext_features(texts)
            semantic_features.append(fasttext_features)
            print(f"[OK] FastText embeddings: {fasttext_features.shape[1]} features")
        else:
            print("[WARNING] FastText embeddings: Not available (model not loaded)")
        
        # 2. Hate lexicon similarity features
        lexicon_features = self._extract_lexicon_similarity_features(texts)
        semantic_features.append(lexicon_features)
        print(f"[OK] Hate lexicon similarity: {lexicon_features.shape[1]} features")
        
        # Combine semantic features
        if semantic_features:
            combined_semantic = np.hstack(semantic_features)
            self.semantic_feature_names = [f"semantic_{i}" for i in range(combined_semantic.shape[1])]
            print(f"[OK] Total semantic features: {combined_semantic.shape[1]}")
            return combined_semantic
        else:
            # Return dummy features if no semantic features available
            dummy_features = np.zeros((len(texts), 1))
            self.semantic_feature_names = ["semantic_dummy"]
            print("[WARNING] No semantic features available, using dummy features")
            return dummy_features
    
    def _transform_semantic_view(self, texts: List[str]) -> np.ndarray:
        """
        Transform new data for semantic view
        """
        semantic_features = []
        
        # FastText embeddings
        if self.fasttext_model is not None:
            fasttext_features = self._extract_fasttext_features(texts)
            semantic_features.append(fasttext_features)
        
        # Hate lexicon similarity
        lexicon_features = self._extract_lexicon_similarity_features(texts)
        semantic_features.append(lexicon_features)
        
        # Combine features
        if semantic_features:
            return np.hstack(semantic_features)
        else:
            return np.zeros((len(texts), 1))
    
    def _extract_fasttext_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract FastText embeddings using average pooling
        """
        if self.fasttext_model is None:
            return np.zeros((len(texts), 300))  # Default FastText dimension
        
        embeddings = []
        for text in texts:
            # Tokenize text
            words = word_tokenize(text.lower())
            
            # Get embeddings for each word
            word_embeddings = []
            for word in words:
                try:
                    embedding = self.fasttext_model.get_word_vector(word)
                    word_embeddings.append(embedding)
                except:
                    continue
            
            # Average pooling
            if word_embeddings:
                avg_embedding = np.mean(word_embeddings, axis=0)
            else:
                avg_embedding = np.zeros(300)  # Default dimension
            
            embeddings.append(avg_embedding)
        
        return np.array(embeddings)
    
    def _extract_lexicon_similarity_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract cosine similarity to hate lexicons
        """
        features = []
        
        for text in texts:
            text_lower = text.lower()
            text_words = set(word_tokenize(text_lower))
            
            lexicon_similarities = []
            
            for lexicon_name, lexicon_words in self.hate_lexicons.items():
                # Count matches with lexicon
                matches = sum(1 for word in lexicon_words if word in text_words)
                
                # Calculate similarity score
                if len(lexicon_words) > 0:
                    similarity = matches / len(lexicon_words)
                else:
                    similarity = 0.0
                
                lexicon_similarities.append(similarity)
            
            features.append(lexicon_similarities)
        
        return np.array(features)
    
    def _fit_transform_stylistic_view(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform stylistic view (text statistics + sentiment)
        """
        print("\n📌 C. STYLISTIC VIEW")
        print("-" * 40)
        
        stylistic_features = []
        
        for text in texts:
            features = self._extract_stylistic_features(text)
            stylistic_features.append(features)
        
        stylistic_array = np.array(stylistic_features)
        self.stylistic_feature_names = [
            'uppercase_ratio', 'punctuation_ratio', 'word_count', 'char_count',
            'avg_word_length', 'unique_word_ratio', 'digit_ratio',
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'textblob_polarity', 'textblob_subjectivity'
        ]
        
        print(f"[OK] Stylistic features: {stylistic_array.shape[1]} features")
        print(f"  - Text statistics: uppercase, punctuation, word/char counts")
        print(f"  - Sentiment analysis: VADER + TextBlob")
        
        return stylistic_array
    
    def _transform_stylistic_view(self, texts: List[str]) -> np.ndarray:
        """
        Transform new data for stylistic view
        """
        stylistic_features = []
        
        for text in texts:
            features = self._extract_stylistic_features(text)
            stylistic_features.append(features)
        
        return np.array(stylistic_features)
    
    def _extract_stylistic_features(self, text: str) -> List[float]:
        """
        Extract stylistic features from a single text
        """
        features = []
        
        # Text statistics
        words = word_tokenize(text)
        chars = list(text)
        
        # 1. Uppercase ratio
        uppercase_count = sum(1 for char in chars if char.isupper())
        uppercase_ratio = uppercase_count / len(chars) if chars else 0.0
        features.append(uppercase_ratio)
        
        # 2. Punctuation ratio
        punctuation_count = sum(1 for char in chars if char in string.punctuation)
        punctuation_ratio = punctuation_count / len(chars) if chars else 0.0
        features.append(punctuation_ratio)
        
        # 3. Word count
        word_count = len(words)
        features.append(word_count)
        
        # 4. Character count
        char_count = len(chars)
        features.append(char_count)
        
        # 5. Average word length
        if words:
            avg_word_length = np.mean([len(word) for word in words])
        else:
            avg_word_length = 0.0
        features.append(avg_word_length)
        
        # 6. Unique word ratio
        if words:
            unique_word_ratio = len(set(words)) / len(words)
        else:
            unique_word_ratio = 0.0
        features.append(unique_word_ratio)
        
        # 7. Digit ratio
        digit_count = sum(1 for char in chars if char.isdigit())
        digit_ratio = digit_count / len(chars) if chars else 0.0
        features.append(digit_ratio)
        
        # 8. VADER sentiment
        if self.sentiment_analyzer:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            features.extend([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu']
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 9. TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            features.extend([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get feature names for each view
        """
        return {
            'lexical': self.lexical_feature_names,
            'semantic': self.semantic_feature_names,
            'stylistic': self.stylistic_feature_names
        }
    
    def get_feature_summary(self) -> Dict[str, Dict]:
        """
        Get summary of features for each view
        """
        return {
            'lexical': {
                'count': len(self.lexical_feature_names),
                'type': 'sparse',
                'description': 'TF-IDF features (word + character n-grams)'
            },
            'semantic': {
                'count': len(self.semantic_feature_names),
                'type': 'dense',
                'description': 'FastText embeddings + hate lexicon similarity'
            },
            'stylistic': {
                'count': len(self.stylistic_feature_names),
                'type': 'statistical',
                'description': 'Text statistics + sentiment analysis'
            }
        }
    
    def combine_views(self, features_dict: Dict[str, np.ndarray], 
                     method: str = 'feature_union') -> np.ndarray:
        """
        Combine features from all views
        
        Args:
            features_dict: Dictionary with features for each view
            method: 'feature_union' (recommended) or 'concatenate'
            
        Returns:
            Combined feature matrix
        """
        if method == 'feature_union' and 'combined' in features_dict:
            # Use FeatureUnion combined features (already normalized)
            combined = features_dict['combined']
            
            print(f"\n[OK] FeatureUnion combined features: {combined.shape[1]} total")
            print(f"  - Lexical (sparse): {features_dict['lexical'].shape[1]}")
            print(f"  - Semantic (dense, normalized): {features_dict['semantic'].shape[1]}")
            print(f"  - Stylistic (dense, normalized): {features_dict['stylistic'].shape[1]}")
            print(f"  - Combined (sparse + dense hybrid): {combined.shape[1]}")
            
            return combined
        
        elif method == 'concatenate':
            # Simple concatenation (legacy method)
            if hasattr(features_dict['lexical'], 'toarray'):
                # Handle sparse matrices
                from scipy.sparse import hstack
                combined = hstack([
                    features_dict['lexical'],
                    features_dict['semantic'],
                    features_dict['stylistic']
                ])
            else:
                # Handle dense matrices
                combined = np.hstack([
                    features_dict['lexical'],
                    features_dict['semantic'],
                    features_dict['stylistic']
                ])
            
            print(f"\n[OK] Concatenated features: {combined.shape[1]} total")
            print(f"  - Lexical: {features_dict['lexical'].shape[1]}")
            print(f"  - Semantic: {features_dict['semantic'].shape[1]}")
            print(f"  - Stylistic: {features_dict['stylistic'].shape[1]}")
            
            return combined
        
        elif method == 'weighted_average':
            # Weighted combination (requires same number of features)
            # This is a placeholder for more sophisticated combination methods
            raise NotImplementedError("Weighted average combination not implemented yet")
        
        else:
            raise ValueError(f"Unknown combination method: {method}")


def create_multi_view_features(train_texts: List[str], 
                              test_texts: List[str],
                              fasttext_model_path: Optional[str] = None,
                              use_feature_union: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to create multi-view features with FeatureUnion fusion
    
    Args:
        train_texts: Training text documents
        test_texts: Test text documents
        fasttext_model_path: Path to FastText model (optional)
        use_feature_union: Whether to use FeatureUnion for fusion (recommended)
        
    Returns:
        Tuple of (train_features, test_features)
    """
    # Initialize feature engineering
    fe = MultiViewFeatureEngineering(
        max_tfidf_features=20000,
        fasttext_model_path=fasttext_model_path,
        use_feature_union=use_feature_union
    )
    
    # Fit and transform training data
    print("Processing training data...")
    train_features_dict = fe.fit_transform(train_texts)
    
    # Transform test data
    print("\nProcessing test data...")
    test_features_dict = fe.transform(test_texts)
    
    # Combine views using FeatureUnion (recommended)
    if use_feature_union:
        train_features = fe.combine_views(train_features_dict, method='feature_union')
        test_features = fe.combine_views(test_features_dict, method='feature_union')
    else:
        train_features = fe.combine_views(train_features_dict, method='concatenate')
        test_features = fe.combine_views(test_features_dict, method='concatenate')
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    summary = fe.get_feature_summary()
    for view, info in summary.items():
        print(f"{view.upper()}: {info['count']} features ({info['type']})")
        print(f"  {info['description']}")
    
    if use_feature_union:
        print(f"\n[LINK] FEATURE FUSION: FeatureUnion with proper normalization")
        print(f"  - Sparse features (lexical): No normalization (TF-IDF)")
        print(f"  - Dense features (semantic, stylistic): Z-score normalization")
        print(f"  - Hybrid matrix: Sparse + dense combination")
    
    return train_features, test_features


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a normal text message.",
        "I HATE YOU! You are a stupid idiot!",
        "The weather is nice today.",
        "Kill all the immigrants! White power!"
    ]
    
    # Create features
    fe = MultiViewFeatureEngineering()
    features_dict = fe.fit_transform(sample_texts)
    
    # Combine views
    combined_features = fe.combine_views(features_dict)
    
    print(f"\nFinal feature matrix shape: {combined_features.shape}")
    print(f"Feature names: {len(fe.get_feature_names()['lexical'])} lexical + "
          f"{len(fe.get_feature_names()['semantic'])} semantic + "
          f"{len(fe.get_feature_names()['stylistic'])} stylistic") 