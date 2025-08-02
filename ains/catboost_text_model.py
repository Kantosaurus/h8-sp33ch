#!/usr/bin/env python3
"""
CatBoost Text Mode for Hate Speech Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

class CatBoostTextModel:
    """
    CatBoost model with built-in text feature transformer
    """
    
    def __init__(self, random_state: int = 42, text_features: List[str] = None):
        """
        Initialize CatBoost text model
        
        Args:
            random_state: Random seed for reproducibility
            text_features: List of text column names to use as text features
        """
        self.random_state = random_state
        self.text_features = text_features or ['text']
        self.is_trained = False
        
        # CatBoost model
        self.model = None
        
        # Text preprocessing results
        self.text_processing_results = {}
        
        # Results storage
        self.results = {}
        
    def _initialize_model(self):
        """
        Initialize CatBoost model with text features
        """
        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_seed=self.random_state,
            eval_metric='F1',
            verbose=100,
            early_stopping_rounds=50,
            use_best_model=True,
            text_features=self.text_features
        )
        
    def _prepare_data(self, texts: List[str], y: np.ndarray = None, 
                     additional_features: Dict[str, np.ndarray] = None) -> Pool:
        """
        Prepare data for CatBoost with text features
        
        Args:
            texts: List of text strings
            y: Target labels (optional for prediction)
            additional_features: Additional numerical/categorical features
            
        Returns:
            CatBoost Pool object
        """
        # Create DataFrame with text features
        data = pd.DataFrame({self.text_features[0]: texts})
        
        # Add additional features if provided
        if additional_features:
            for feature_name, feature_values in additional_features.items():
                data[feature_name] = feature_values
        
        # Create CatBoost Pool
        if y is not None:
            pool = Pool(data=data, label=y, text_features=self.text_features)
        else:
            pool = Pool(data=data, text_features=self.text_features)
            
        return pool
    
    def _extract_text_statistics(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract basic text statistics as additional features
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with text statistics
        """
        print("Extracting text statistics...")
        
        stats = {
            'text_length': [],
            'word_count': [],
            'avg_word_length': [],
            'uppercase_ratio': [],
            'punctuation_count': [],
            'digit_count': [],
            'special_char_count': []
        }
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Text length
            stats['text_length'].append(len(text))
            
            # Word count
            words = text.split()
            stats['word_count'].append(len(words))
            
            # Average word length
            if words:
                avg_word_len = np.mean([len(word) for word in words])
            else:
                avg_word_len = 0
            stats['avg_word_length'].append(avg_word_len)
            
            # Uppercase ratio
            if text:
                uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            else:
                uppercase_ratio = 0
            stats['uppercase_ratio'].append(uppercase_ratio)
            
            # Punctuation count
            punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            punct_count = sum(1 for c in text if c in punctuation)
            stats['punctuation_count'].append(punct_count)
            
            # Digit count
            digit_count = sum(1 for c in text if c.isdigit())
            stats['digit_count'].append(digit_count)
            
            # Special character count (non-alphanumeric, non-space)
            special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
            stats['special_char_count'].append(special_count)
        
        # Convert to numpy arrays
        for key in stats:
            stats[key] = np.array(stats[key])
        
        print(f"✓ Extracted {len(stats)} text statistics")
        return stats
    
    def train(self, texts: List[str], y: np.ndarray, 
             additional_features: Dict[str, np.ndarray] = None,
             validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train CatBoost model with text features
        
        Args:
            texts: List of text strings
            y: Target labels
            additional_features: Additional numerical/categorical features
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results
        """
        print("=" * 60)
        print("TRAINING CATBOOST TEXT MODEL")
        print("=" * 60)
        
        # Initialize model
        self._initialize_model()
        
        # Extract text statistics
        text_stats = self._extract_text_statistics(texts)
        
        # Combine additional features
        all_additional_features = text_stats.copy()
        if additional_features:
            all_additional_features.update(additional_features)
        
        # Split data
        X_train_texts, X_val_texts, y_train, y_val = train_test_split(
            texts, y, test_size=validation_split, random_state=self.random_state, stratify=y
        )
        
        # Prepare training data
        train_pool = self._prepare_data(X_train_texts, y_train, all_additional_features)
        
        # Prepare validation data
        val_pool = self._prepare_data(X_val_texts, y_val, all_additional_features)
        
        # Train model
        print(f"\nTraining CatBoost with {len(self.text_features)} text features...")
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            plot=False
        )
        
        # Evaluate on validation data
        val_predictions = self.model.predict(val_pool)
        val_probabilities = self.model.predict_proba(val_pool)[:, 1]
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions)
        val_precision = precision_score(y_val, val_predictions)
        val_recall = recall_score(y_val, val_predictions)
        val_auc = roc_auc_score(y_val, val_probabilities)
        
        # Store results
        self.results = {
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc
        }
        
        # Store text processing results
        self.text_processing_results = {
            'text_stats': text_stats,
            'n_texts_processed': len(texts),
            'text_features_used': self.text_features
        }
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        
        self.is_trained = True
        
        return self.results
    
    def predict(self, texts: List[str], 
               additional_features: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            texts: List of text strings
            additional_features: Additional numerical/categorical features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract text statistics
        text_stats = self._extract_text_statistics(texts)
        
        # Combine additional features
        all_additional_features = text_stats.copy()
        if additional_features:
            all_additional_features.update(additional_features)
        
        # Prepare data
        pool = self._prepare_data(texts, additional_features=all_additional_features)
        
        # Make predictions
        predictions = self.model.predict(pool)
        
        return predictions
    
    def predict_proba(self, texts: List[str], 
                     additional_features: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            texts: List of text strings
            additional_features: Additional numerical/categorical features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract text statistics
        text_stats = self._extract_text_statistics(texts)
        
        # Combine additional features
        all_additional_features = text_stats.copy()
        if additional_features:
            all_additional_features.update(additional_features)
        
        # Prepare data
        pool = self._prepare_data(texts, additional_features=all_additional_features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(pool)
        
        return probabilities
    
    def evaluate(self, texts: List[str], y: np.ndarray,
                additional_features: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            texts: List of text strings
            y: True labels
            additional_features: Additional numerical/categorical features
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(texts, additional_features)
        probabilities = self.predict_proba(texts, additional_features)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        auc = roc_auc_score(y, probabilities)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def analyze_text_features(self) -> pd.DataFrame:
        """
        Analyze text feature processing
        
        Returns:
            DataFrame with text feature analysis
        """
        if not self.text_processing_results:
            raise ValueError("No text processing results available. Train the model first.")
        
        text_stats = self.text_processing_results['text_stats']
        
        # Calculate statistics for each text feature
        stats = []
        for feature_name, values in text_stats.items():
            stats.append({
                'Feature': feature_name,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Median': np.median(values)
            })
        
        return pd.DataFrame(stats)
    
    def cross_validate(self, texts: List[str], y: np.ndarray,
                      additional_features: Dict[str, np.ndarray] = None,
                      cv_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation
        
        Args:
            texts: List of text strings
            y: Target labels
            additional_features: Additional numerical/categorical features
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
            print(f"  Fold {fold + 1}/{cv_folds}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Prepare additional features for this fold
            train_additional = None
            val_additional = None
            if additional_features:
                train_additional = {k: v[train_idx] for k, v in additional_features.items()}
                val_additional = {k: v[val_idx] for k, v in additional_features.items()}
            
            # Train model
            model = CatBoostTextModel(random_state=self.random_state, text_features=self.text_features)
            model.train(train_texts, y_train, train_additional)
            
            # Evaluate
            fold_results = model.evaluate(val_texts, y_val, val_additional)
            
            # Store results
            for metric in cv_results:
                cv_results[metric].append(fold_results[metric])
        
        # Calculate mean and std
        summary = {}
        for metric, values in cv_results.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        
        print(f"\nCross-validation Results:")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            mean_val = summary[f'{metric}_mean']
            std_val = summary[f'{metric}_std']
            print(f"  {metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return summary


def create_catboost_text_model(texts: List[str], y: np.ndarray,
                              additional_features: Dict[str, np.ndarray] = None,
                              random_state: int = 42) -> CatBoostTextModel:
    """
    Convenience function to create and train a CatBoost text model
    
    Args:
        texts: List of text strings
        y: Target labels
        additional_features: Additional numerical/categorical features
        random_state: Random seed
        
    Returns:
        Trained CatBoostTextModel instance
    """
    print("=" * 60)
    print("CREATING CATBOOST TEXT MODEL")
    print("=" * 60)
    
    # Initialize model
    model = CatBoostTextModel(random_state=random_state)
    
    # Train model
    training_results = model.train(texts, y, additional_features)
    
    print("\n" + "=" * 60)
    print("CATBOOST TEXT MODEL CREATED!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("CatBoost Text Model Example")
    print("=" * 50)
    
    # Create dummy text data
    np.random.seed(42)
    n_samples = 100
    
    # Dummy texts
    dummy_texts = [
        "This is a hateful comment that should be detected",
        "I love this community and everyone here",
        "You are all terrible people and should leave",
        "Great discussion, thanks for sharing",
        "I hate you all and wish you would disappear",
        "This is a wonderful place to be",
        "You're all idiots and don't deserve to be here",
        "Thanks for the helpful information",
        "I despise everything about this",
        "What a fantastic experience this has been"
    ] * (n_samples // 10)
    
    # Add some random texts
    import random
    words = ["hate", "love", "terrible", "great", "awful", "amazing", "horrible", "wonderful", "disgusting", "beautiful"]
    for _ in range(n_samples - len(dummy_texts)):
        text = " ".join(random.choices(words, k=random.randint(3, 8)))
        dummy_texts.append(text)
    
    # Dummy labels (some correlation with hate words)
    dummy_labels = np.array([1 if any(word in text.lower() for word in ["hate", "terrible", "awful", "horrible", "disgusting"]) else 0 
                            for text in dummy_texts])
    
    # Additional features
    additional_features = {
        'text_length': np.array([len(text) for text in dummy_texts]),
        'word_count': np.array([len(text.split()) for text in dummy_texts])
    }
    
    # Create and train model
    model = create_catboost_text_model(dummy_texts, dummy_labels, additional_features)
    
    # Make predictions
    predictions = model.predict(dummy_texts, additional_features)
    probabilities = model.predict_proba(dummy_texts, additional_features)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Analyze feature importance
    importance_df = model.get_feature_importance()
    print(f"\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Analyze text features
    text_analysis = model.analyze_text_features()
    print(f"\nText feature analysis:")
    print(text_analysis.to_string(index=False))
    
    # Cross-validation
    cv_results = model.cross_validate(dummy_texts, dummy_labels, additional_features) 