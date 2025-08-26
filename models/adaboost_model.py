import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AdaBoostModel:
    """
    AdaBoost model for hate speech classification
    Adaptive Boosting with decision tree base estimators
    """
    
    def __init__(self, random_state=42):
        # Create base estimator (decision tree)
        base_estimator = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state
        )
        
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=100,
            learning_rate=1.0,
            random_state=random_state
        )
        self.is_trained = False
        self.cv_scores = None
        
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train the AdaBoost model with cross-validation
        """
        print("Training AdaBoost...")
        
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
        
        print(f"AdaBoost CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
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
        
        print("AdaBoost Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return importance
    
    def get_estimator_weights(self):
        """
        Get weights of individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting estimator weights")
        
        return self.model.estimator_weights_
    
    def get_estimator_errors(self):
        """
        Get errors of individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting estimator errors")
        
        return self.model.estimator_errors_ 

if __name__ == "__main__":
    from data_utils import load_and_preprocess_data
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Create and train model
    model = AdaBoostModel()
    cv_results = model.train(X_train, y_train)
    
    # Evaluate model
    test_results = model.evaluate(X_test, y_test)
    
    print("\nAdaBoost Model Training Complete!")