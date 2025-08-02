import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class SVMModel:
    """
    Support Vector Machine with linear kernel for hate speech classification
    """
    
    def __init__(self, random_state=42):
        self.model = LinearSVC(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            loss='squared_hinge'
        )
        self.is_trained = False
        self.cv_scores = None
        
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train the SVM model with cross-validation
        """
        print("Training SVM (LinearSVC)...")
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='f1')
        
        self.cv_scores = {
            'f1_mean': cv_scores.mean(),
            'f1_std': cv_scores.std(),
            'f1_scores': cv_scores
        }
        
        # Train on full dataset
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"SVM CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.cv_scores
    
    def predict(self, X):
        """
        Make binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using decision function
        Note: LinearSVC doesn't have predict_proba, so we use decision_function
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get decision function scores
        decision_scores = self.model.decision_function(X)
        
        # Convert to probabilities using sigmoid-like transformation
        # This is an approximation since LinearSVC doesn't provide probabilities
        proba = 1 / (1 + np.exp(-decision_scores))
        
        # Return as 2D array [prob_class_0, prob_class_1]
        return np.column_stack([1 - proba, proba])
    
    def decision_function(self, X):
        """
        Get raw decision function scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.decision_function(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("SVM Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = np.abs(self.model.coef_[0])
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return importance 