import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class CatBoostModel:
    """
    CatBoost model for hate speech classification
    Gradient boosting with categorical features support
    """
    
    def __init__(self, random_state=42):
        self.model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=random_state,
            verbose=False,
            early_stopping_rounds=10,
            use_best_model=True
        )
        self.is_trained = False
        self.cv_scores = None
        
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train the CatBoost model with cross-validation
        """
        print("Training CatBoost...")
        
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
        
        print(f"CatBoost CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("CatBoost Results:")
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
    
    def get_feature_importance_type(self, importance_type='PredictionValuesChange'):
        """
        Get feature importance with different types
        Types: 'PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance'
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.model.get_feature_importance(type=importance_type)
    
    def get_best_iteration(self):
        """
        Get the best iteration number
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting best iteration")
        
        return self.model.get_best_iteration()
    
    def get_evals_result(self):
        """
        Get evaluation results during training
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting evaluation results")
        
        return self.model.get_evals_result()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        self.model.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}") 