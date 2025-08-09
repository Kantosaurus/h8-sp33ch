import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight

# ML Models for different "layers"
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                            GradientBoostingClassifier, AdaBoostClassifier,
                            VotingClassifier, BaggingClassifier)
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, 
                                SGDClassifier, PassiveAggressiveClassifier)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Advanced models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# Text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    Serializable identity transformer that returns input unchanged.
    Used as a fallback when other transformers fail to fit.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X


class DeepTextFeatureExtractor:
    """
    Input Layer: Comprehensive text feature extraction mimicking input neurons
    """
    
    def __init__(self, max_features=20000):
        self.max_features = max_features
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Multiple vectorizers for diverse feature views optimized for hate speech
        self.tfidf_word = TfidfVectorizer(
            max_features=max_features//4, ngram_range=(1,3), 
            stop_words='english', sublinear_tf=True,
            min_df=2, max_df=0.95  # Filter out very rare and very common words
        )
        self.tfidf_char = TfidfVectorizer(
            max_features=max_features//4, analyzer='char', 
            ngram_range=(2,6), stop_words='english',  # Increased to capture hate patterns
            min_df=2, max_df=0.95
        )
        self.count_vec = CountVectorizer(
            max_features=max_features//4, ngram_range=(1,3),  # Increased for better context
            stop_words='english', min_df=2, max_df=0.95
        )
        self.hash_vec = HashingVectorizer(
            n_features=max_features//4, ngram_range=(1,3),  # Increased for better patterns
            alternate_sign=False
        )
        
    def fit(self, texts):
        """Fit all vectorizers"""
        self.tfidf_word.fit(texts)
        self.tfidf_char.fit(texts)
        self.count_vec.fit(texts)
        return self
        
    def transform(self, texts):
        """Transform texts into comprehensive feature matrix"""
        # Traditional text features
        tfidf_word_features = self.tfidf_word.transform(texts)
        tfidf_char_features = self.tfidf_char.transform(texts)
        count_features = self.count_vec.transform(texts)
        hash_features = self.hash_vec.transform(texts)
        
        # Statistical features
        stat_features = []
        for text in texts:
            features = self._extract_statistical_features(text)
            stat_features.append(features)
        
        stat_features = np.array(stat_features)
        
        # Combine all features
        from scipy.sparse import hstack, csr_matrix
        combined = hstack([
            tfidf_word_features,
            tfidf_char_features, 
            count_features,
            hash_features,
            csr_matrix(stat_features)
        ])
        
        return combined
    
    def _extract_statistical_features(self, text):
        """Extract enhanced statistical features from text for hate speech detection"""
        features = []
        text_lower = text.lower()
        
        # Basic stats
        features.append(len(text))  # Character count
        features.append(len(text.split()))  # Word count
        features.append(len([s for s in text.split('.') if s.strip()]))  # Sentence count
        
        # Capitalization (important for hate speech - often uses caps)
        features.append(sum(1 for c in text if c.isupper()) / max(len(text), 1))
        features.append(sum(1 for c in text if c.islower()) / max(len(text), 1))
        features.append(sum(1 for word in text.split() if word.isupper()) / max(len(text.split()), 1))  # ALL CAPS words
        
        # Punctuation (hate speech often uses excessive punctuation)
        features.append(sum(1 for c in text if c in string.punctuation) / max(len(text), 1))
        features.append(text.count('!') / max(len(text), 1))
        features.append(text.count('?') / max(len(text), 1))
        features.append(text.count('!!!') / max(len(text), 1))  # Multiple exclamation marks
        features.append(text.count('???') / max(len(text), 1))  # Multiple question marks
        
        # Hate speech specific patterns
        hate_indicators = ['hate', 'kill', 'die', 'stupid', 'idiot', 'moron', 'dumb', 'ugly', 'fat']
        features.append(sum(1 for word in hate_indicators if word in text_lower) / max(len(text.split()), 1))
        
        # Profanity and offensive language indicators (simplified)
        offensive_chars = ['@', '#', '*', '%']  # Often used to mask profanity
        features.append(sum(1 for c in offensive_chars if c in text) / max(len(text), 1))
        
        # Repetitive characters (e.g., "sooooo", "hateeeee")
        import re
        repeated_chars = len(re.findall(r'(.)\1{2,}', text_lower))
        features.append(repeated_chars / max(len(text), 1))
        
        # Sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features.extend([sentiment['compound'], sentiment['pos'], 
                        sentiment['neg'], sentiment['neu']])
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            features.extend([blob.sentiment.polarity, blob.sentiment.subjectivity])
        except:
            features.extend([0.0, 0.0])
        
        # Repetition features
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            features.append(unique_ratio)
            
            # Average word length (hate speech sometimes uses shorter, angrier words)
            avg_word_length = sum(len(word) for word in words) / len(words)
            features.append(avg_word_length)
        else:
            features.extend([0.0, 0.0])
        
        # Social media patterns
        features.append(text.count('@') / max(len(text), 1))  # Mentions
        features.append(text.count('#') / max(len(text), 1))  # Hashtags
        
        return features


class HiddenLayer:
    """
    Hidden Layer: Simulates a neural network hidden layer using ensemble models
    Each layer transforms input features through multiple models and applies activation
    """
    
    def __init__(self, layer_id, n_models=5, random_state=42):
        self.layer_id = layer_id
        self.n_models = n_models
        self.random_state = random_state
        self.models = {}
        self.feature_transformers = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_models(self):
        """Create diverse models for this layer with class balancing"""
        models = [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=self.random_state, 
                                        class_weight='balanced', n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=self.random_state, 
                                    eval_metric='logloss', scale_pos_weight=1.82)),  # 13033/7151 ratio
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=500, class_weight='balanced')),
            ('svc', SVC(probability=True, random_state=self.random_state, kernel='rbf', class_weight='balanced')),
            ('nb', GaussianNB()),  # No class_weight parameter
            ('knn', KNeighborsClassifier(n_neighbors=5)),  # No class_weight parameter
            ('et', ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=self.random_state, 
                                      class_weight='balanced', n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),  # No class_weight in older versions
            ('cb', cb.CatBoostClassifier(iterations=50, random_state=self.random_state, verbose=False,
                                       class_weights=[1, 1.82])),  # Balanced for minority class
            ('lgb', lgb.LGBMClassifier(n_estimators=50, random_state=self.random_state, verbose=-1,
                                     class_weight='balanced'))
        ]
        
        # Select n_models randomly
        np.random.seed(self.random_state + self.layer_id)
        selected = np.random.choice(len(models), self.n_models, replace=False)
        
        for i, idx in enumerate(selected):
            name, model = models[idx]
            self.models[f"{name}_{i}"] = model
    
    def _create_feature_transformers(self, input_dim):
        """Create feature transformation techniques"""
        # Dimensionality reduction techniques
        self.feature_transformers = {
            'pca': PCA(n_components=min(100, input_dim//2)),
            'svd': TruncatedSVD(n_components=min(50, input_dim//4)),
            'select_k': SelectKBest(f_classif, k=min(200, input_dim//2))
        }
    
    def fit(self, X, y):
        """Fit the hidden layer"""
        print(f"Training Hidden Layer {self.layer_id}...")
        
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X_dense)
        
        # Create feature transformers
        self._create_feature_transformers(X_scaled.shape[1])
        
        # Fit feature transformers
        for name, transformer in self.feature_transformers.items():
            try:
                transformer.fit(X_scaled)
            except:
                # If transformer fails, create identity transformer
                self.feature_transformers[name] = IdentityTransformer()
        
        # Compute sample weights for class balancing
        sample_weights = compute_sample_weight('balanced', y)
        
        # Create and train models
        self._create_models()
        
        for name, model in self.models.items():
            try:
                # Use different feature views for different models
                if 'pca' in name:
                    X_transformed = self.feature_transformers['pca'].transform(X_scaled)
                elif 'svd' in name:
                    X_transformed = self.feature_transformers['svd'].transform(X_scaled)
                elif 'select' in name:
                    X_transformed = self.feature_transformers['select_k'].transform(X_scaled)
                else:
                    X_transformed = X_scaled
                
                # Try to fit with sample weights first, fallback to regular fit
                try:
                    if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                        model.fit(X_transformed, y, sample_weight=sample_weights)
                    else:
                        model.fit(X_transformed, y)
                except TypeError:
                    # Some models don't support sample_weight
                    model.fit(X_transformed, y)
                    
                print(f"  ✓ Trained {name}")
            except Exception as e:
                print(f"  ✗ Failed to train {name}: {e}")
                # Remove failed model
                del self.models[name]
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform input through the layer (like forward pass)"""
        if not self.is_fitted:
            raise ValueError("Layer must be fitted before transform")
        
        # Handle sparse matrices
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
            
        X_scaled = self.scaler.transform(X_dense)
        
        # Get predictions from all models (like neuron outputs)
        layer_outputs = []
        
        for name, model in self.models.items():
            try:
                # Use appropriate feature transformation
                if 'pca' in name:
                    X_transformed = self.feature_transformers['pca'].transform(X_scaled)
                elif 'svd' in name:
                    X_transformed = self.feature_transformers['svd'].transform(X_scaled)
                elif 'select' in name:
                    X_transformed = self.feature_transformers['select_k'].transform(X_scaled)
                else:
                    X_transformed = X_scaled
                
                # Get probability predictions (continuous outputs like neurons)
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_transformed)
                    layer_outputs.append(probs[:, 1])  # Probability of positive class
                else:
                    preds = model.predict(X_transformed)
                    layer_outputs.append(preds.astype(float))
                    
            except Exception as e:
                print(f"Warning: Model {name} failed in transform: {e}")
                # Add zeros if model fails
                layer_outputs.append(np.zeros(X_scaled.shape[0]))
        
        # Apply activation function (non-linear transformation)
        activations = np.column_stack(layer_outputs)
        return self._apply_activation(activations)
    
    def _apply_activation(self, x):
        """Apply activation function (simulating neural network activation)"""
        # Multiple activation functions for diversity
        activations = []
        
        # ReLU-like
        activations.append(np.maximum(0, x))
        
        # Sigmoid-like
        activations.append(1 / (1 + np.exp(-np.clip(x, -500, 500))))
        
        # Tanh-like
        activations.append(np.tanh(x))
        
        # Leaky ReLU-like
        activations.append(np.where(x > 0, x, 0.01 * x))
        
        # Combine activations
        combined = np.concatenate(activations, axis=1)
        return combined


class DeepEnsembleClassifier:
    """
    Main Deep Neural Network-like Ensemble Classifier
    
    Architecture:
    Input Layer → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Output Layer
    
    Each layer uses different ML models to simulate neurons
    Includes dropout-like regularization and weight optimization
    """
    
    def __init__(self, n_hidden_layers=3, models_per_layer=6, random_state=42, 
                 focus_on_minority_class=True):
        self.n_hidden_layers = n_hidden_layers
        self.models_per_layer = models_per_layer
        self.random_state = random_state
        self.focus_on_minority_class = focus_on_minority_class  # Aggressive focus on hate speech detection
        
        # Network components with enhanced feature extraction for hate speech
        self.feature_extractor = DeepTextFeatureExtractor(max_features=30000)  # More features for complexity
        self.hidden_layers = []
        self.output_layer = None
        self.final_scaler = StandardScaler()
        self.optimal_threshold = 0.5  # Will be optimized during training
        
        # Training history
        self.training_history = {
            'layer_scores': [],
            'final_score': None,
            'optimal_threshold': None,
            'class_distribution': None
        }
        
    def _create_hidden_layers(self):
        """Create hidden layers"""
        for i in range(self.n_hidden_layers):
            layer = HiddenLayer(
                layer_id=i, 
                n_models=self.models_per_layer,
                random_state=self.random_state + i
            )
            self.hidden_layers.append(layer)
    
    def _create_output_layer(self):
        """Create output layer (final ensemble) with class balancing"""
        # Use best performing models for output with class balancing
        output_models = [
            ('rf_out', RandomForestClassifier(n_estimators=200, random_state=self.random_state, 
                                            class_weight='balanced', n_jobs=-1)),
            ('xgb_out', xgb.XGBClassifier(n_estimators=200, random_state=self.random_state, 
                                        eval_metric='logloss', scale_pos_weight=1.82)),
            ('cb_out', cb.CatBoostClassifier(iterations=200, random_state=self.random_state, 
                                           verbose=False, class_weights=[1, 1.82])),
            ('lgb_out', lgb.LGBMClassifier(n_estimators=200, random_state=self.random_state, 
                                         verbose=-1, class_weight='balanced')),
            ('lr_out', LogisticRegression(random_state=self.random_state, max_iter=1000, 
                                        class_weight='balanced'))
        ]
        
        self.output_layer = VotingClassifier(
            estimators=output_models,
            voting='soft',
            n_jobs=-1
        )
    
    def fit(self, X_text, y):
        """Train the deep ensemble (like training a neural network)"""
        print("=" * 60)
        print("TRAINING DEEP NEURAL NETWORK-LIKE ENSEMBLE")
        print("=" * 60)
        
        # Input layer: Extract comprehensive features
        print("Input Layer: Extracting features...")
        self.feature_extractor.fit(X_text)
        X_features = self.feature_extractor.transform(X_text)
        print(f"Input features shape: {X_features.shape}")
        
        # Create network architecture
        self._create_hidden_layers()
        self._create_output_layer()
        
        # Forward pass through hidden layers
        current_input = X_features
        
        for i, layer in enumerate(self.hidden_layers):
            print(f"\nHidden Layer {i+1}:")
            layer.fit(current_input, y)
            current_input = layer.transform(current_input)
            print(f"Hidden Layer {i+1} output shape: {current_input.shape}")
            
            # Evaluate layer performance (like monitoring training loss)
            layer_score = self._evaluate_layer_output(current_input, y)
            self.training_history['layer_scores'].append(layer_score)
            print(f"Hidden Layer {i+1} internal score: {layer_score:.4f}")
        
        # Final scaling before output layer
        current_input = self.final_scaler.fit_transform(current_input)
        
        # Train output layer with sample weights
        print(f"\nOutput Layer:")
        output_sample_weights = compute_sample_weight('balanced', y)
        
        # Fit output layer with sample weights where possible
        try:
            self.output_layer.fit(current_input, y, sample_weight=output_sample_weights)
        except TypeError:
            # VotingClassifier doesn't support sample_weight directly
            self.output_layer.fit(current_input, y)
        
        # Optimize decision threshold for macro F1
        print(f"\nOptimizing decision threshold...")
        self.optimal_threshold = self._optimize_threshold(current_input, y)
        self.training_history['optimal_threshold'] = self.optimal_threshold
        print(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        # Final evaluation with optimized threshold
        final_probas = self.output_layer.predict_proba(current_input)[:, 1]
        final_pred = (final_probas >= self.optimal_threshold).astype(int)
        final_score = f1_score(y, final_pred, average='macro')
        self.training_history['final_score'] = final_score
        
        print(f"\nTraining Complete!")
        print(f"Final Ensemble Score (with optimized threshold): {final_score:.4f}")
        print("=" * 60)
        
        return self
    
    def _optimize_threshold(self, X_val, y_val):
        """Optimize decision threshold for macro F1 score"""
        probas = self.output_layer.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (probas >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"  Best threshold: {best_threshold:.3f} with macro F1: {best_f1:.4f}")
        return best_threshold
    
    def _evaluate_layer_output(self, layer_output, y_true):
        """Evaluate intermediate layer output"""
        # Use a simple classifier to evaluate layer representations
        from sklearn.linear_model import LogisticRegression
        temp_clf = LogisticRegression(random_state=self.random_state, max_iter=500, class_weight='balanced')
        
        # Split for evaluation
        X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
            layer_output, y_true, test_size=0.2, random_state=self.random_state, stratify=y_true
        )
        
        temp_clf.fit(X_temp_train, y_temp_train)
        temp_pred = temp_clf.predict(X_temp_val)
        return f1_score(y_temp_val, temp_pred, average='macro')
    
    def predict(self, X_text):
        """Make predictions (forward pass) with optimized threshold"""
        # Forward pass through the network
        X_features = self.feature_extractor.transform(X_text)
        
        current_input = X_features
        for layer in self.hidden_layers:
            current_input = layer.transform(current_input)
        
        current_input = self.final_scaler.transform(current_input)
        
        # Use optimized threshold for predictions
        probas = self.output_layer.predict_proba(current_input)[:, 1]
        return (probas >= self.optimal_threshold).astype(int)
    
    def predict_proba(self, X_text):
        """Get prediction probabilities"""
        X_features = self.feature_extractor.transform(X_text)
        
        current_input = X_features
        for layer in self.hidden_layers:
            current_input = layer.transform(current_input)
        
        current_input = self.final_scaler.transform(current_input)
        return self.output_layer.predict_proba(current_input)
    
    def get_network_summary(self):
        """Get summary of the network architecture"""
        summary = {
            'architecture': f"Input → {self.n_hidden_layers} Hidden Layers → Output",
            'models_per_layer': self.models_per_layer,
            'training_scores': self.training_history['layer_scores'],
            'final_score': self.training_history['final_score']
        }
        return summary


def train_deep_ensemble(train_data_path, test_data_path, random_state=42):
    """
    Train the deep ensemble classifier
    """
    print("Loading data...")
    
    # Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    print(f"Training data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Prepare training data
    # Filter out any header rows that might be mixed in the data
    train_data = train_data[train_data['label'] != 'label']
    
    # Convert labels to integers
    train_data['label'] = pd.to_numeric(train_data['label'], errors='coerce')
    
    # Remove any rows with invalid labels
    train_data = train_data.dropna(subset=['label'])
    train_data['label'] = train_data['label'].astype(int)
    
    X_train_text = train_data['post'].fillna('').values
    y_train = train_data['label'].values
    X_test_text = test_data['post'].fillna('').values
    
    class_dist = np.bincount(y_train)
    print(f"Class distribution: {class_dist}")
    print(f"Class imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
    
    # Create and train the deep ensemble with aggressive hate speech focus
    deep_ensemble = DeepEnsembleClassifier(
        n_hidden_layers=4,  # Increased layers for more complexity
        models_per_layer=8,  # More models per layer for better diversity
        random_state=random_state,
        focus_on_minority_class=True
    )
    
    # Train the model
    deep_ensemble.fit(X_train_text, y_train)
    
    # Make predictions
    print("\nGenerating predictions...")
    test_predictions = deep_ensemble.predict(X_test_text)
    test_probabilities = deep_ensemble.predict_proba(X_test_text)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_data['id'],
        'label': test_predictions
    })
    
    # Save results
    submission.to_csv('deep_ensemble_predictions.csv', index=False)
    
    # Network summary
    summary = deep_ensemble.get_network_summary()
    print(f"\nNetwork Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nPredictions saved to: deep_ensemble_predictions.csv")
    
    return deep_ensemble, submission


if __name__ == "__main__":
    # Train the deep ensemble
    train_path = "../data/combined.csv"
    test_path = "../data/test.csv"
    
    deep_ensemble, submission = train_deep_ensemble(train_path, test_path)