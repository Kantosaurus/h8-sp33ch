import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class LogisticMetaClassifier:
    """
    Logistic Regression meta-classifier using base models' probabilities and confidence features
    
    This meta-classifier combines:
    1. Predicted probabilities from view-specialized models
    2. Confidence scores (margin between top two probabilities)
    3. Additional meta-features for robust ensemble learning
    """
    
    def __init__(self, random_state: int = 42, C: float = 1.0, class_weight: str = 'balanced'):
        self.random_state = random_state
        self.C = C
        self.class_weight = class_weight
        
        # Initialize components
        self.meta_classifier = LogisticRegression(
            random_state=random_state,
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            solver='liblinear'
        )
        self.scaler = StandardScaler()
        
        # Training state
        self.is_trained = False
        self.feature_names = []
        self.results = {}
        
        # Store view-specialized models reference
        self.view_specialized_models = None
        
        # Threshold optimization
        self.optimal_threshold = 0.5  # Default threshold
        self.threshold_optimization_results = {}
    
    def set_view_specialized_models(self, view_specialized_models):
        """
        Set reference to view-specialized models for feature extraction
        
        Args:
            view_specialized_models: ViewSpecializedModels instance
        """
        self.view_specialized_models = view_specialized_models
        print("[OK] View-specialized models linked to meta-classifier")
    
    def extract_meta_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract comprehensive meta-features from view-specialized models
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Meta-features matrix
        """
        if self.view_specialized_models is None:
            raise ValueError("View-specialized models not set. Call set_view_specialized_models() first.")
        
        if not self.view_specialized_models.is_trained:
            raise ValueError("View-specialized models not trained")
        
        meta_features = []
        
        # 1. Get predictions from view-specialized models
        predictions = self.view_specialized_models.predict_views(features_dict)
        
        # 2. Get confidence scores
        confidence_scores = self.view_specialized_models.get_confidence_scores(features_dict)
        
        # 3. Extract base meta-features
        base_meta_features = self.view_specialized_models.extract_view_meta_features(features_dict)
        
        # 4. Add predictions (probabilities)
        for view_name in ['lexical', 'semantic', 'stylistic']:
            if view_name in predictions:
                meta_features.append(predictions[view_name])
        
        # 5. Add confidence scores
        for view_name in ['lexical', 'semantic', 'stylistic']:
            if view_name in confidence_scores:
                meta_features.append(confidence_scores[view_name])
        
        # 6. Add base meta-features
        meta_features.append(base_meta_features)
        
        # 7. Calculate additional meta-features
        additional_features = self._calculate_additional_meta_features(predictions, confidence_scores)
        meta_features.append(additional_features)
        
        # Combine all meta-features
        meta_features_array = np.column_stack(meta_features)
        
        return meta_features_array
    
    def _calculate_additional_meta_features(self, predictions: Dict[str, np.ndarray], 
                                         confidence_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate additional meta-features for enhanced learning
        
        Args:
            predictions: Dictionary with predictions from each view
            confidence_scores: Dictionary with confidence scores from each view
            
        Returns:
            Additional meta-features array
        """
        additional_features = []
        
        # Convert to arrays for easier computation
        pred_arrays = []
        conf_arrays = []
        
        for view_name in ['lexical', 'semantic', 'stylistic']:
            if view_name in predictions:
                pred_arrays.append(predictions[view_name])
                conf_arrays.append(confidence_scores[view_name])
        
        if not pred_arrays:
            return np.array([])
        
        pred_matrix = np.column_stack(pred_arrays)
        conf_matrix = np.column_stack(conf_arrays)
        
        # 1. Mean prediction across views
        mean_prediction = np.mean(pred_matrix, axis=1, keepdims=True)
        additional_features.append(mean_prediction)
        
        # 2. Standard deviation of predictions (model disagreement)
        pred_std = np.std(pred_matrix, axis=1, keepdims=True)
        additional_features.append(pred_std)
        
        # 3. Maximum prediction
        max_prediction = np.max(pred_matrix, axis=1, keepdims=True)
        additional_features.append(max_prediction)
        
        # 4. Minimum prediction
        min_prediction = np.min(pred_matrix, axis=1, keepdims=True)
        additional_features.append(min_prediction)
        
        # 5. Prediction range
        pred_range = max_prediction - min_prediction
        additional_features.append(pred_range)
        
        # 6. Mean confidence across views
        mean_confidence = np.mean(conf_matrix, axis=1, keepdims=True)
        additional_features.append(mean_confidence)
        
        # 7. Standard deviation of confidence
        conf_std = np.std(conf_matrix, axis=1, keepdims=True)
        additional_features.append(conf_std)
        
        # 8. Maximum confidence
        max_confidence = np.max(conf_matrix, axis=1, keepdims=True)
        additional_features.append(max_confidence)
        
        # 9. Minimum confidence
        min_confidence = np.min(conf_matrix, axis=1, keepdims=True)
        additional_features.append(min_confidence)
        
        # 10. Confidence range
        conf_range = max_confidence - min_confidence
        additional_features.append(conf_range)
        
        # 11. Number of models with high confidence (>0.8)
        high_conf_count = np.sum(conf_matrix > 0.8, axis=1, keepdims=True)
        additional_features.append(high_conf_count)
        
        # 12. Number of models with low confidence (<0.2)
        low_conf_count = np.sum(conf_matrix < 0.2, axis=1, keepdims=True)
        additional_features.append(low_conf_count)
        
        # 13. Agreement indicator (models predicting same class)
        pred_classes = (pred_matrix > 0.5).astype(int)
        agreement = np.std(pred_classes, axis=1, keepdims=True) == 0
        additional_features.append(agreement.astype(float))
        
        # 14. Prediction variance
        pred_variance = np.var(pred_matrix, axis=1, keepdims=True)
        additional_features.append(pred_variance)
        
        # 15. Confidence variance
        conf_variance = np.var(conf_matrix, axis=1, keepdims=True)
        additional_features.append(conf_variance)
        
        return np.column_stack(additional_features)
    
    def optimize_threshold(self, y_val: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Find optimal threshold that maximizes F1-score
        
        Args:
            y_val: True labels for validation set
            y_proba: Predicted probabilities for validation set
            
        Returns:
            Optimal threshold value
        """
        print("Optimizing threshold to maximize F1-score...")
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        threshold_scores = []
        
        for thresh in thresholds:
            y_pred = (y_proba > thresh).astype(int)
            f1 = f1_score(y_val, y_pred)
            threshold_scores.append((thresh, f1))
        
        # Find best threshold
        best_thresh, best_f1 = max(threshold_scores, key=lambda x: x[1])
        
        # Store results
        self.threshold_optimization_results = {
            'optimal_threshold': best_thresh,
            'best_f1_score': best_f1,
            'all_thresholds': thresholds,
            'all_f1_scores': [score[1] for score in threshold_scores]
        }
        
        # Update optimal threshold
        self.optimal_threshold = best_thresh
        
        print(f"[OK] Optimal threshold: {best_thresh:.3f} (F1-score: {best_f1:.4f})")
        
        return best_thresh
    
    def train(self, features_dict: Dict[str, np.ndarray], y: np.ndarray, 
             cv_folds: int = 5, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the Logistic Regression meta-classifier
        
        Args:
            features_dict: Dictionary with features for each view
            y: Target labels
            cv_folds: Number of cross-validation folds
            validation_split: Fraction for validation set
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "=" * 60)
        print("TRAINING LOGISTIC REGRESSION META-CLASSIFIER")
        print("=" * 60)
        
        # Extract meta-features
        print("Extracting meta-features...")
        meta_features = self.extract_meta_features(features_dict)
        
        print(f"Meta-features shape: {meta_features.shape}")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            meta_features, y, test_size=validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        # Scale features
        print("Scaling meta-features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Cross-validation
        print(f"Performing {cv_folds}-fold cross-validation...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_accuracy = cross_val_score(self.meta_classifier, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(self.meta_classifier, X_train_scaled, y_train, cv=cv, scoring='f1')
        cv_precision = cross_val_score(self.meta_classifier, X_train_scaled, y_train, cv=cv, scoring='precision')
        cv_recall = cross_val_score(self.meta_classifier, X_train_scaled, y_train, cv=cv, scoring='recall')
        cv_auc = cross_val_score(self.meta_classifier, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        
        # Train final model
        print("Training final meta-classifier...")
        self.meta_classifier.fit(X_train_scaled, y_train)
        
        # Validation performance
        y_val_pred = self.meta_classifier.predict(X_val_scaled)
        y_val_pred_proba = self.meta_classifier.predict_proba(X_val_scaled)[:, 1]
        
        # Optimize threshold
        self.optimize_threshold(y_val, y_val_pred_proba)
        
        # Use optimal threshold for validation predictions
        y_val_pred_optimal = (y_val_pred_proba > self.optimal_threshold).astype(int)
        
        val_accuracy = accuracy_score(y_val, y_val_pred_optimal)
        val_f1 = f1_score(y_val, y_val_pred_optimal)
        val_precision = precision_score(y_val, y_val_pred_optimal)
        val_recall = recall_score(y_val, y_val_pred_optimal)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        # Store results
        self.results = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_precision_mean': cv_precision.mean(),
            'cv_precision_std': cv_precision.std(),
            'cv_recall_mean': cv_recall.mean(),
            'cv_recall_std': cv_recall.std(),
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_auc': val_auc,
            'meta_features_shape': meta_features.shape
        }
        
        self.is_trained = True
        
        # Print results
        print(f"\nCross-validation results ({cv_folds} folds):")
        print(f"  Accuracy: {self.results['cv_accuracy_mean']:.4f} ± {self.results['cv_accuracy_std']:.4f}")
        print(f"  F1-Score: {self.results['cv_f1_mean']:.4f} ± {self.results['cv_f1_std']:.4f}")
        print(f"  Precision: {self.results['cv_precision_mean']:.4f} ± {self.results['cv_precision_std']:.4f}")
        print(f"  Recall: {self.results['cv_recall_mean']:.4f} ± {self.results['cv_recall_std']:.4f}")
        print(f"  AUC: {self.results['cv_auc_mean']:.4f} ± {self.results['cv_auc_std']:.4f}")
        
        print(f"\nValidation results:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        
        return self.results
    
    def predict(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained meta-classifier with optimal threshold
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before making predictions")
        
        # Get probabilities
        probabilities = self.predict_proba(features_dict)[:, 1]
        
        # Apply optimal threshold
        predictions = (probabilities > self.optimal_threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities from the meta-classifier
        
        Args:
            features_dict: Dictionary with features for each view
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before making predictions")
        
        # Extract meta-features
        meta_features = self.extract_meta_features(features_dict)
        
        # Scale features
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Get probabilities
        probabilities = self.meta_classifier.predict_proba(meta_features_scaled)
        
        return probabilities
    
    def evaluate(self, features_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the meta-classifier
        
        Args:
            features_dict: Dictionary with features for each view
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(features_dict)
        y_pred_proba = self.predict_proba(features_dict)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        
        print(f"\nMeta-classifier evaluation results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the meta-classifier
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before getting feature importance")
        
        # Get coefficients (absolute values for importance)
        importance = np.abs(self.meta_classifier.coef_[0])
        
        return importance
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the meta-features
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # View predictions
        for view in ['lexical', 'semantic', 'stylistic']:
            feature_names.append(f"{view}_prediction")
        
        # View confidence scores
        for view in ['lexical', 'semantic', 'stylistic']:
            feature_names.append(f"{view}_confidence")
        
        # Base meta-features (from view-specialized models)
        base_feature_count = 6  # 3 predictions + 3 confidence scores
        for i in range(base_feature_count):
            feature_names.append(f"base_meta_{i}")
        
        # Additional meta-features
        additional_features = [
            'mean_prediction', 'pred_std', 'max_prediction', 'min_prediction', 'pred_range',
            'mean_confidence', 'conf_std', 'max_confidence', 'min_confidence', 'conf_range',
            'high_conf_count', 'low_conf_count', 'agreement', 'pred_variance', 'conf_variance'
        ]
        feature_names.extend(additional_features)
        
        return feature_names
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze and display feature importance
        
        Returns:
            DataFrame with feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before analyzing feature importance")
        
        importance = self.get_feature_importance()
        feature_names = self.get_feature_names()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\n" + "=" * 60)
        print("META-CLASSIFIER FEATURE IMPORTANCE")
        print("=" * 60)
        print(importance_df.to_string(index=False, float_format='%.4f'))
        
        return importance_df
    
    def analyze_threshold_optimization(self) -> pd.DataFrame:
        """
        Analyze threshold optimization results
        
        Returns:
            DataFrame with threshold optimization analysis
        """
        if not self.threshold_optimization_results:
            raise ValueError("Threshold optimization not performed. Train the model first.")
        
        # Create DataFrame with all threshold results
        threshold_df = pd.DataFrame({
            'Threshold': self.threshold_optimization_results['all_thresholds'],
            'F1_Score': self.threshold_optimization_results['all_f1_scores']
        })
        
        # Add optimal threshold info
        optimal_thresh = self.threshold_optimization_results['optimal_threshold']
        optimal_f1 = self.threshold_optimization_results['best_f1_score']
        
        print("\n" + "=" * 60)
        print("THRESHOLD OPTIMIZATION ANALYSIS")
        print("=" * 60)
        print(f"Optimal threshold: {optimal_thresh:.3f}")
        print(f"Best F1-score: {optimal_f1:.4f}")
        print(f"Default threshold (0.5) F1-score: {threshold_df[threshold_df['Threshold'] == 0.5]['F1_Score'].iloc[0]:.4f}")
        print(f"Improvement: {optimal_f1 - threshold_df[threshold_df['Threshold'] == 0.5]['F1_Score'].iloc[0]:.4f}")
        
        # Show top 10 thresholds
        print(f"\nTop 10 thresholds by F1-score:")
        top_thresholds = threshold_df.nlargest(10, 'F1_Score')
        print(top_thresholds.to_string(index=False, float_format='%.4f'))
        
        return threshold_df
    
    def plot_threshold_optimization(self):
        """
        Plot threshold optimization results
        """
        if not self.threshold_optimization_results:
            raise ValueError("Threshold optimization not performed. Train the model first.")
        
        try:
            import matplotlib.pyplot as plt
            
            thresholds = self.threshold_optimization_results['all_thresholds']
            f1_scores = self.threshold_optimization_results['all_f1_scores']
            optimal_thresh = self.threshold_optimization_results['optimal_threshold']
            optimal_f1 = self.threshold_optimization_results['best_f1_score']
            
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
            plt.axvline(x=optimal_thresh, color='r', linestyle='--', 
                       label=f'Optimal Threshold: {optimal_thresh:.3f}')
            plt.axvline(x=0.5, color='g', linestyle='--', 
                       label='Default Threshold: 0.5')
            plt.axhline(y=optimal_f1, color='r', linestyle=':', alpha=0.7,
                       label=f'Best F1: {optimal_f1:.4f}')
            
            plt.xlabel('Threshold')
            plt.ylabel('F1 Score')
            plt.title('Threshold Optimization for F1 Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
    
    
def create_logistic_meta_classifier(view_specialized_models, 
                                  features_dict: Dict[str, np.ndarray],
                                  y: np.ndarray,
                                  random_state: int = 42) -> LogisticMetaClassifier:
    """
    Convenience function to create and train a Logistic Regression meta-classifier
    
    Args:
        view_specialized_models: Trained ViewSpecializedModels instance
        features_dict: Dictionary with features for each view
        y: Target labels
        random_state: Random seed
        
    Returns:
        Trained LogisticMetaClassifier instance
    """
    print("=" * 60)
    print("CREATING LOGISTIC REGRESSION META-CLASSIFIER")
    print("=" * 60)
    
    # Initialize meta-classifier
    meta_classifier = LogisticMetaClassifier(random_state=random_state)
    
    # Set view-specialized models
    meta_classifier.set_view_specialized_models(view_specialized_models)
    
    # Train meta-classifier
    training_results = meta_classifier.train(features_dict, y)
    
    print("\n" + "=" * 60)
    print("LOGISTIC META-CLASSIFIER CREATED!")
    print("=" * 60)
    
    return meta_classifier


if __name__ == "__main__":
    # Example usage
    print("Logistic Regression Meta-Classifier Example")
    print("=" * 50)
    
    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Dummy features for each view
    dummy_features = {
        'lexical': np.random.rand(n_samples, 1000),
        'semantic': np.random.rand(n_samples, 300),
        'stylistic': np.random.rand(n_samples, 14)
    }
    
    # Dummy labels
    dummy_labels = np.random.randint(0, 2, n_samples)
    
    # Create view-specialized models (mock)
    from view_specialized_models import ViewSpecializedModels
    vsm = ViewSpecializedModels(random_state=42)
    vsm.train_models(dummy_features, dummy_labels)
    
    # Create and train meta-classifier
    meta_classifier = create_logistic_meta_classifier(vsm, dummy_features, dummy_labels)
    
    # Make predictions
    predictions = meta_classifier.predict(dummy_features)
    probabilities = meta_classifier.predict_proba(dummy_features)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Analyze feature importance
    importance_df = meta_classifier.analyze_feature_importance() 