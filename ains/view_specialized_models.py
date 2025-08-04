import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# CatBoost for semantic features
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

class ViewSpecializedModels:
    """
    Model specialization with per-view learners
    
    Each feature view gets its own specialized model:
    - Lexical: Logistic Regression (great on sparse binary)
    - Semantic: CatBoost (handles dense features, text-aware)
    - Stylistic: Random Forest (nonlinear, handles discretes well)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.is_trained = False
        self.results = {}
        
        # Initialize specialized models for each view
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize specialized models for each feature view
        """
        print("Initializing view-specialized models...")
        
        # Lexical View: Logistic Regression (great on sparse binary)
        self.models['lexical'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=2000,  # Increased for better convergence
            solver='liblinear',  # Good for sparse data
            C=0.1,  # Reduced for better regularization
            class_weight='balanced',
            tol=1e-4  # Tighter tolerance
        )
        
        # Semantic View: CatBoost (handles dense features, text-aware)
        if CATBOOST_AVAILABLE:
            self.models['semantic'] = CatBoostClassifier(
                random_state=self.random_state,
                iterations=500,
                learning_rate=0.1,
                depth=6,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                task_type='CPU'
            )
        else:
            # Fallback to Random Forest if CatBoost not available
            print("[WARNING] CatBoost not available, using Random Forest for semantic view")
            self.models['semantic'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        
        # Stylistic View: Random Forest (nonlinear, handles discretes well)
        self.models['stylistic'] = RandomForestClassifier(
            n_estimators=200,  # Increased for better performance
            max_depth=10,  # Slightly increased
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,  # Use all CPU cores
            bootstrap=True,
            oob_score=True  # Out-of-bag score for validation
        )
        
        print("[OK] View-specialized models initialized:")
        print(f"  - Lexical: Logistic Regression (sparse binary)")
        print(f"  - Semantic: {'CatBoost' if CATBOOST_AVAILABLE else 'Random Forest'} (dense features)")
        print(f"  - Stylistic: Random Forest (nonlinear, discretes)")
    
    def train_models(self, features_dict: Dict[str, np.ndarray], y: np.ndarray, 
                    cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Train specialized models for each view
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results for each view
        """
        print("\n" + "=" * 60)
        print("TRAINING VIEW-SPECIALIZED MODELS")
        print("=" * 60)
        
        results = {}
        
        for view_name, features in features_dict.items():
            if view_name == 'combined':  # Skip combined features
                continue
                
            print(f"\n📌 Training {view_name.upper()} model...")
            print("-" * 40)
            
            # Get the specialized model for this view
            model = self.models[view_name]
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                X = features.toarray()
            else:
                X = features
            
            print(f"Features shape: {X.shape}")
            
            # Cross-validation
            cv_scores = self._cross_validate_model(model, X, y, cv_folds, view_name)
            
            # Train final model
            print("Training final model...")
            model.fit(X, y)
            
            # Store results
            results[view_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'feature_shape': X.shape
            }
            
            print(f"[OK] {view_name} model trained successfully")
        
        self.is_trained = True
        self.results = results
        
        return results
    
    def _cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                            cv_folds: int, view_name: str) -> Dict[str, float]:
        """
        Perform cross-validation for a model
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
        cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        cv_scores = {
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'precision_mean': cv_precision.mean(),
            'precision_std': cv_precision.std(),
            'recall_mean': cv_recall.mean(),
            'recall_std': cv_recall.std(),
            'auc_mean': cv_auc.mean(),
            'auc_std': cv_auc.std()
        }
        
        print(f"Cross-validation results ({cv_folds} folds):")
        print(f"  Accuracy: {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f}")
        print(f"  F1-Score: {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
        print(f"  Precision: {cv_scores['precision_mean']:.4f} ± {cv_scores['precision_std']:.4f}")
        print(f"  Recall: {cv_scores['recall_mean']:.4f} ± {cv_scores['recall_std']:.4f}")
        print(f"  AUC: {cv_scores['auc_mean']:.4f} ± {cv_scores['auc_std']:.4f}")
        
        return cv_scores
    
    def predict_views(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get predictions from all view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Dictionary with predictions for each view
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        
        for view_name, features in features_dict.items():
            if view_name == 'combined':  # Skip combined features
                continue
                
            model = self.models[view_name]
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                X = features.toarray()
            else:
                X = features
            
            # Get predictions
            pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
            predictions[view_name] = pred_proba
        
        return predictions
    
    def get_confidence_scores(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get confidence scores (margin between top two probabilities) for each view
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Dictionary with confidence scores for each view
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before getting confidence scores")
        
        confidence_scores = {}
        
        for view_name, features in features_dict.items():
            if view_name == 'combined':  # Skip combined features
                continue
                
            model = self.models[view_name]
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                X = features.toarray()
            else:
                X = features
            
            # Get prediction probabilities
            pred_proba = model.predict_proba(X)
            
            # Calculate confidence score (margin between top two probabilities)
            # For binary classification: |prob_class_1 - prob_class_0|
            confidence = np.abs(pred_proba[:, 1] - pred_proba[:, 0])
            confidence_scores[view_name] = confidence
        
        return confidence_scores
    
    def extract_view_meta_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract meta-features from view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Meta-features matrix with predictions and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before extracting meta-features")
        
        # Get predictions and confidence scores
        predictions = self.predict_views(features_dict)
        confidence_scores = self.get_confidence_scores(features_dict)
        
        # Combine into meta-features
        meta_features = []
        
        for view_name in ['lexical', 'semantic', 'stylistic']:
            if view_name in predictions:
                # Add predicted probabilities
                meta_features.append(predictions[view_name])
                
                # Add confidence scores
                meta_features.append(confidence_scores[view_name])
        
        # Stack all meta-features
        meta_features_array = np.column_stack(meta_features)
        
        return meta_features_array
    
    def evaluate_models(self, features_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            y: True labels
            
        Returns:
            Dictionary with evaluation results for each view
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        print("\n" + "=" * 60)
        print("EVALUATING VIEW-SPECIALIZED MODELS")
        print("=" * 60)
        
        evaluation_results = {}
        
        for view_name, features in features_dict.items():
            if view_name == 'combined':  # Skip combined features
                continue
                
            print(f"\n📌 Evaluating {view_name.upper()} model...")
            print("-" * 40)
            
            model = self.models[view_name]
            
            # Handle sparse matrices
            if hasattr(features, 'toarray'):
                X = features.toarray()
            else:
                X = features
            
            # Get predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            
            # Calculate confidence scores
            confidence_scores = self.get_confidence_scores({view_name: features})
            avg_confidence = np.mean(confidence_scores[view_name])
            
            results = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'avg_confidence': avg_confidence
            }
            
            evaluation_results[view_name] = results
            
            print(f"Test Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Avg Confidence: {avg_confidence:.4f}")
        
        return evaluation_results
    
    def compare_view_performance(self, evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare performance across all views
        
        Args:
            evaluation_results: Results from evaluate_models()
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for view_name, results in evaluation_results.items():
            comparison_data.append({
                'View': view_name.title(),
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'AUC': results['auc'],
                'Avg Confidence': results['avg_confidence']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 60)
        print("VIEW-SPECIALIZED MODEL COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def get_feature_importance(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get feature importance from view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Dictionary with feature importance for each view
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before getting feature importance")
        
        importance_dict = {}
        
        for view_name, features in features_dict.items():
            if view_name == 'combined':  # Skip combined features
                continue
                
            model = self.models[view_name]
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_dict[view_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importance_dict[view_name] = np.abs(model.coef_[0])
            else:
                print(f"Warning: No feature importance available for {view_name} model")
        
        return importance_dict


def create_view_specialized_ensemble(features_dict: Dict[str, np.ndarray], 
                                   y: np.ndarray,
                                   random_state: int = 42) -> ViewSpecializedModels:
    """
    Convenience function to create and train view-specialized models
    
    Args:
        features_dict: Dictionary with features for each view
        y: Target labels
        random_state: Random seed
        
    Returns:
        Trained ViewSpecializedModels instance
    """
    print("=" * 60)
    print("CREATING VIEW-SPECIALIZED ENSEMBLE")
    print("=" * 60)
    
    # Initialize view-specialized models
    vsm = ViewSpecializedModels(random_state=random_state)
    
    # Train models
    training_results = vsm.train_models(features_dict, y)
    
    # Evaluate models
    evaluation_results = vsm.evaluate_models(features_dict, y)
    
    # Compare performance
    comparison_df = vsm.compare_view_performance(evaluation_results)
    
    print("\n" + "=" * 60)
    print("VIEW-SPECIALIZED ENSEMBLE CREATED!")
    print("=" * 60)
    
    return vsm


if __name__ == "__main__":
    # Example usage
    print("View-Specialized Models Example")
    print("=" * 50)
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Dummy features for each view
    dummy_features = {
        'lexical': np.random.rand(n_samples, 1000),  # Sparse-like
        'semantic': np.random.rand(n_samples, 300),  # Dense
        'stylistic': np.random.rand(n_samples, 14)   # Dense
    }
    
    # Dummy labels
    dummy_labels = np.random.randint(0, 2, n_samples)
    
    # Create view-specialized ensemble
    vsm = create_view_specialized_ensemble(dummy_features, dummy_labels)
    
    # Extract meta-features
    meta_features = vsm.extract_view_meta_features(dummy_features)
    print(f"\nMeta-features shape: {meta_features.shape}")
    
    # Get confidence scores
    confidence_scores = vsm.get_confidence_scores(dummy_features)
    for view, scores in confidence_scores.items():
        print(f"{view} confidence range: [{scores.min():.3f}, {scores.max():.3f}]") 