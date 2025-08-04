#!/usr/bin/env python3
"""
Boosted Stacking Ensemble with Meta-Features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class BoostedStackingEnsemble:
    """
    Boosted Stacking Ensemble with advanced meta-features
    """
    
    def __init__(self, random_state: int = 42, n_folds: int = 5):
        """
        Initialize the boosted stacking ensemble
        
        Args:
            random_state: Random seed for reproducibility
            n_folds: Number of cross-validation folds for stacking
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.is_trained = False
        
        # Base models for different feature spaces
        self.base_models = {}
        
        # Meta-model
        self.meta_model = None
        
        # Meta-features storage
        self.meta_features_train = None
        self.meta_features_test = None
        
        # Results storage
        self.results = {}
        
    def _initialize_base_models(self):
        """
        Initialize diverse base models for different feature spaces
        """
        self.base_models = {
            'lexical_lr': {
                'model': RidgeClassifier(random_state=self.random_state),
                'feature_space': 'lexical',
                'description': 'Ridge Classifier for lexical features'
            },
            'semantic_xgb': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'feature_space': 'semantic',
                'description': 'XGBoost for semantic features'
            },
            'stylistic_rf': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.random_state
                ),
                'feature_space': 'stylistic',
                'description': 'Random Forest for stylistic features'
            },
            'hybrid_gb': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state
                ),
                'feature_space': 'hybrid',
                'description': 'Gradient Boosting for hybrid features'
            },
            'svm_linear': {
                'model': LinearSVC(
                    random_state=self.random_state,
                    max_iter=2000,
                    dual=False,
                    class_weight='balanced'
                ),
                'feature_space': 'hybrid',
                'description': 'Linear SVM for hybrid features with decision function'
            }
        }
        
    def _initialize_meta_model(self):
        """
        Initialize meta-model (XGBoost for better generalization)
        """
        self.meta_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
    def _extract_meta_features(self, features_dict: Dict[str, np.ndarray], 
                             y: np.ndarray = None, is_training: bool = True) -> np.ndarray:
        """
        Extract advanced meta-features from base models
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels (for training)
            is_training: Whether this is training or prediction
            
        Returns:
            Meta-features matrix
        """
        print("Extracting advanced meta-features...")
        
        # Initialize cross-validation for stacking
        if is_training:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        meta_features_list = []
        
        for fold_idx, (model_name, model_info) in enumerate(self.base_models.items()):
            print(f"  Processing {model_name}: {model_info['description']}")
            
            model = model_info['model']
            feature_space = model_info['feature_space']
            
            # Get appropriate features
            if feature_space == 'lexical':
                features = features_dict.get('lexical', np.zeros((features_dict['semantic'].shape[0], 1000)))
            elif feature_space == 'semantic':
                features = features_dict.get('semantic', np.zeros((features_dict['lexical'].shape[0], 300)))
            elif feature_space == 'stylistic':
                features = features_dict.get('stylistic', np.zeros((features_dict['lexical'].shape[0], 14)))
            elif feature_space == 'hybrid':
                # Combine all features for hybrid model
                # Get the number of samples from the first available feature
                n_samples = None
                for key in ['lexical', 'semantic', 'stylistic']:
                    if key in features_dict:
                        if hasattr(features_dict[key], 'shape'):
                            n_samples = features_dict[key].shape[0]
                            break
                        else:
                            n_samples = len(features_dict[key])
                            break
                
                if n_samples is None:
                    raise ValueError("No features found in features_dict")
                
                # Ensure all features are 2D and have the same number of samples
                lexical_features = features_dict.get('lexical', np.zeros((n_samples, 1000)))
                semantic_features = features_dict.get('semantic', np.zeros((n_samples, 300)))
                stylistic_features = features_dict.get('stylistic', np.zeros((n_samples, 14)))
                
                # Handle sparse matrices and ensure 2D
                if hasattr(lexical_features, 'toarray'):
                    lexical_features = lexical_features.toarray()
                if hasattr(semantic_features, 'toarray'):
                    semantic_features = semantic_features.toarray()
                if hasattr(stylistic_features, 'toarray'):
                    stylistic_features = stylistic_features.toarray()
                
                # Ensure 2D arrays
                if lexical_features.ndim == 1:
                    lexical_features = lexical_features.reshape(-1, 1)
                if semantic_features.ndim == 1:
                    semantic_features = semantic_features.reshape(-1, 1)
                if stylistic_features.ndim == 1:
                    stylistic_features = stylistic_features.reshape(-1, 1)
                
                # Ensure correct number of samples
                if lexical_features.shape[0] != n_samples:
                    lexical_features = lexical_features[:n_samples]
                if semantic_features.shape[0] != n_samples:
                    semantic_features = semantic_features[:n_samples]
                if stylistic_features.shape[0] != n_samples:
                    stylistic_features = stylistic_features[:n_samples]
                
                features = np.hstack([lexical_features, semantic_features, stylistic_features])
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                features = features.toarray()
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            if is_training:
                # Cross-validation for stacking
                fold_predictions = np.zeros(len(y))
                fold_probabilities = np.zeros(len(y))
                
                for train_idx, val_idx in skf.split(features, y):
                    X_train_fold, X_val_fold = features[train_idx], features[val_idx]
                    y_train_fold = y[train_idx]
                    
                    # Train model on fold
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation fold
                    fold_predictions[val_idx] = model.predict(X_val_fold)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        fold_probabilities[val_idx] = model.predict_proba(X_val_fold)[:, 1]
                    elif hasattr(model, 'decision_function'):
                        # For LinearSVC, use decision function as probability proxy
                        decision_scores = model.decision_function(X_val_fold)
                        # Convert to probability-like scores using sigmoid
                        fold_probabilities[val_idx] = 1 / (1 + np.exp(-decision_scores))
                    else:
                        fold_probabilities[val_idx] = model.predict(X_val_fold)
                
                # Train final model on full data
                model.fit(features, y)
                
                # Store predictions and probabilities
                meta_features_list.extend([
                    fold_predictions,  # Model predictions
                    fold_probabilities,  # Model probabilities
                ])
                
            else:
                # Prediction mode
                predictions = model.predict(features)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[:, 1]
                elif hasattr(model, 'decision_function'):
                    # For LinearSVC, use decision function as probability proxy
                    decision_scores = model.decision_function(features)
                    # Convert to probability-like scores using sigmoid
                    probabilities = 1 / (1 + np.exp(-decision_scores))
                else:
                    probabilities = predictions
                
                meta_features_list.extend([
                    predictions,
                    probabilities
                ])
        
        # Combine all meta-features
        meta_features = np.column_stack(meta_features_list)
        
        # Add advanced meta-features
        advanced_features = self._extract_advanced_meta_features(meta_features, is_training)
        meta_features = np.column_stack([meta_features, advanced_features])
        
        # Add decision function features
        decision_features = self._extract_decision_functions(features_dict, is_training)
        meta_features = np.column_stack([meta_features, decision_features])
        
        print(f"[OK] Extracted {meta_features.shape[1]} meta-features (including decision functions)")
        return meta_features
    
    def _extract_advanced_meta_features(self, base_meta_features: np.ndarray, 
                                      is_training: bool) -> np.ndarray:
        """
        Extract advanced meta-features: disagreement, confidence, etc.
        
        Args:
            base_meta_features: Base meta-features from models
            is_training: Whether this is training or prediction
            
        Returns:
            Advanced meta-features
        """
        n_samples = base_meta_features.shape[0]
        n_models = len(self.base_models)
        
        # Reshape to separate predictions and probabilities
        predictions = base_meta_features[:, :n_models]
        probabilities = base_meta_features[:, n_models:2*n_models]
        
        advanced_features = []
        
        # 1. Model disagreement (variance across predictions)
        prediction_variance = np.var(predictions, axis=1, ddof=1)
        advanced_features.append(prediction_variance)
        
        # 2. Probability variance
        probability_variance = np.var(probabilities, axis=1, ddof=1)
        advanced_features.append(probability_variance)
        
        # 3. Confidence scores (prob_max - prob_2nd_max)
        sorted_probs = np.sort(probabilities, axis=1)
        confidence = sorted_probs[:, -1] - sorted_probs[:, -2]
        advanced_features.append(confidence)
        
        # 4. Agreement indicators
        unanimous_agreement = (np.std(predictions, axis=1) == 0).astype(float)
        advanced_features.append(unanimous_agreement)
        
        # 5. Majority vote
        majority_vote = (np.mean(predictions, axis=1) > 0.5).astype(float)
        advanced_features.append(majority_vote)
        
        # 6. Prediction entropy (uncertainty measure)
        prob_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        advanced_features.append(prob_entropy)
        
        # 7. Model agreement with majority
        majority_agreement = np.mean(predictions == majority_vote[:, np.newaxis], axis=1)
        advanced_features.append(majority_agreement)
        
        # 8. High confidence indicators
        high_confidence = (confidence > 0.3).astype(float)
        advanced_features.append(high_confidence)
        
        # 9. Low confidence indicators
        low_confidence = (confidence < 0.1).astype(float)
        advanced_features.append(low_confidence)
        
        # 10. Prediction consistency (how many models predict the same class)
        most_common_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 1, predictions
        )
        consistency = np.mean(predictions == most_common_pred[:, np.newaxis], axis=1)
        advanced_features.append(consistency)
        
        return np.column_stack(advanced_features)
    
    def _extract_decision_functions(self, features_dict: Dict[str, np.ndarray], 
                                  is_training: bool = True) -> np.ndarray:
        """
        Extract decision function values from models that support it
        
        Args:
            features_dict: Dictionary with features for each view
            is_training: Whether this is training or prediction
            
        Returns:
            Decision function values matrix
        """
        decision_functions = []
        
        for model_name, model_info in self.base_models.items():
            model = model_info['model']
            feature_space = model_info['feature_space']
            
            # Get appropriate features
            if feature_space == 'lexical':
                features = features_dict.get('lexical', np.zeros((features_dict['semantic'].shape[0], 1000)))
            elif feature_space == 'semantic':
                features = features_dict.get('semantic', np.zeros((features_dict['lexical'].shape[0], 300)))
            elif feature_space == 'stylistic':
                features = features_dict.get('stylistic', np.zeros((features_dict['lexical'].shape[0], 14)))
            elif feature_space == 'hybrid':
                # Combine all features for hybrid model
                features = np.hstack([
                    features_dict.get('lexical', np.zeros((features_dict['semantic'].shape[0], 1000))),
                    features_dict.get('semantic', np.zeros((features_dict['lexical'].shape[0], 300))),
                    features_dict.get('stylistic', np.zeros((features_dict['lexical'].shape[0], 14)))
                ])
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                features = features.toarray()
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            # Extract decision function if available
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(features)
                decision_functions.append(decision_scores)
            else:
                # Use prediction as fallback
                predictions = model.predict(features)
                decision_functions.append(predictions.astype(float))
        
        return np.column_stack(decision_functions)
    
    def train(self, features_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
        """
        Train the boosted stacking ensemble
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
            
        Returns:
            Training results
        """
        print("=" * 60)
        print("TRAINING BOOSTED STACKING ENSEMBLE")
        print("=" * 60)
        
        # Initialize models
        self._initialize_base_models()
        self._initialize_meta_model()
        
        # Extract meta-features
        self.meta_features_train = self._extract_meta_features(features_dict, y, is_training=True)
        
        # Train meta-model
        print("\nTraining meta-model (XGBoost)...")
        self.meta_model.fit(self.meta_features_train, y)
        
        # Evaluate on training data
        train_predictions = self.meta_model.predict(self.meta_features_train)
        train_probabilities = self.meta_model.predict_proba(self.meta_features_train)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y, train_predictions)
        train_f1 = f1_score(y, train_predictions)
        train_precision = precision_score(y, train_predictions)
        train_recall = recall_score(y, train_predictions)
        train_auc = roc_auc_score(y, train_probabilities)
        
        # Store results
        self.results = {
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_auc': train_auc
        }
        
        print(f"\nTraining Results:")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  F1-Score: {train_f1:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  AUC: {train_auc:.4f}")
        
        self.is_trained = True
        
        return self.results
    
    def predict(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained ensemble
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Extract meta-features
        self.meta_features_test = self._extract_meta_features(features_dict, is_training=False)
        
        # Make predictions
        predictions = self.meta_model.predict(self.meta_features_test)
        
        return predictions
    
    def predict_proba(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Extract meta-features
        self.meta_features_test = self._extract_meta_features(features_dict, is_training=False)
        
        # Get probabilities
        probabilities = self.meta_model.predict_proba(self.meta_features_test)
        
        return probabilities
    
    def evaluate(self, features_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble
        
        Args:
            features_dict: Dictionary with features for each view
            y: True labels
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")
        
        # Make predictions
        predictions = self.predict(features_dict)
        probabilities = self.predict_proba(features_dict)[:, 1]
        
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
        Get feature importance from meta-model
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before getting feature importance")
        
        if hasattr(self.meta_model, 'feature_importances_'):
            importance = self.meta_model.feature_importances_
        else:
            importance = np.abs(self.meta_model.coef_[0])
        
        # Create feature names
        feature_names = []
        
        # Base model features
        for model_name in self.base_models.keys():
            feature_names.extend([f"{model_name}_pred", f"{model_name}_prob"])
        
        # Advanced meta-features
        advanced_names = [
            'prediction_variance', 'probability_variance', 'confidence',
            'unanimous_agreement', 'majority_vote', 'prediction_entropy',
            'majority_agreement', 'high_confidence', 'low_confidence', 'consistency'
        ]
        feature_names.extend(advanced_names)
        
        # Decision function features
        for model_name in self.base_models.keys():
            feature_names.append(f"{model_name}_decision_function")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def analyze_meta_features(self) -> pd.DataFrame:
        """
        Analyze meta-features distribution
        
        Returns:
            DataFrame with meta-features analysis
        """
        if self.meta_features_train is None:
            raise ValueError("No meta-features available. Train the model first.")
        
        # Create feature names
        feature_names = []
        for model_name in self.base_models.keys():
            feature_names.extend([f"{model_name}_pred", f"{model_name}_prob"])
        
        advanced_names = [
            'prediction_variance', 'probability_variance', 'confidence',
            'unanimous_agreement', 'majority_vote', 'prediction_entropy',
            'majority_agreement', 'high_confidence', 'low_confidence', 'consistency'
        ]
        feature_names.extend(advanced_names)
        
        # Decision function features
        for model_name in self.base_models.keys():
            feature_names.append(f"{model_name}_decision_function")
        
        # Calculate statistics
        stats = []
        for i, name in enumerate(feature_names):
            if i < self.meta_features_train.shape[1]:
                values = self.meta_features_train[:, i]
                stats.append({
                    'Feature': name,
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Median': np.median(values)
                })
        
        return pd.DataFrame(stats)


def create_boosted_stacking_ensemble(features_dict: Dict[str, np.ndarray], 
                                   y: np.ndarray, 
                                   random_state: int = 42) -> BoostedStackingEnsemble:
    """
    Convenience function to create and train a boosted stacking ensemble
    
    Args:
        features_dict: Dictionary with features for each view
        y: Target labels
        random_state: Random seed
        
    Returns:
        Trained BoostedStackingEnsemble instance
    """
    print("=" * 60)
    print("CREATING BOOSTED STACKING ENSEMBLE")
    print("=" * 60)
    
    # Initialize ensemble
    ensemble = BoostedStackingEnsemble(random_state=random_state)
    
    # Train ensemble
    training_results = ensemble.train(features_dict, y)
    
    print("\n" + "=" * 60)
    print("BOOSTED STACKING ENSEMBLE CREATED!")
    print("=" * 60)
    
    return ensemble


if __name__ == "__main__":
    # Example usage
    print("Boosted Stacking Ensemble Example")
    print("=" * 50)
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 200
    
    # Dummy features for each view
    dummy_features = {
        'lexical': np.random.rand(n_samples, 1000),
        'semantic': np.random.rand(n_samples, 300),
        'stylistic': np.random.rand(n_samples, 14)
    }
    
    # Dummy labels
    dummy_labels = np.random.randint(0, 2, n_samples)
    
    # Create and train ensemble
    ensemble = create_boosted_stacking_ensemble(dummy_features, dummy_labels)
    
    # Make predictions
    predictions = ensemble.predict(dummy_features)
    probabilities = ensemble.predict_proba(dummy_features)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Analyze feature importance
    importance_df = ensemble.get_feature_importance()
    print(f"\nTop 10 most important meta-features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Analyze meta-features
    meta_analysis = ensemble.analyze_meta_features()
    print(f"\nMeta-features analysis:")
    print(meta_analysis.to_string(index=False)) 