import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class LightGBMModel:
    """
    LightGBM model for hate speech classification
    Fast gradient boosting with leaf-wise tree growth
    """
    
    def __init__(self, random_state=42):
        self.model = LGBMClassifier(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            verbose=-1
        )
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the LightGBM model with optional hyperparameter tuning
        """
        print("Training LightGBM...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            grid_search = GridSearchCV(
                LGBMClassifier(random_state=42, verbose=-1),
                param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        
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
        
        print(f"LightGBM CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("LightGBM Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names=None, importance_type='gain'):
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
    
    def get_feature_importance_split(self, feature_names=None):
        """
        Get feature importance based on split count
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_(importance_type='split')
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'split_importance': importance
            }).sort_values('split_importance', ascending=False)
            return importance_df
        else:
            return importance
    
    def get_model_info(self):
        """
        Get detailed model information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting model info")
        
        return {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'num_leaves': self.model.num_leaves,
            'min_child_samples': self.model.min_child_samples,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'reg_alpha': self.model.reg_alpha,
            'reg_lambda': self.model.reg_lambda,
            'best_params': self.best_params
        }
    
    def get_booster_info(self):
        """
        Get information about the underlying booster
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting booster info")
        
        booster = self.model.booster_
        return {
            'num_trees': booster.num_trees(),
            'num_features': booster.num_features(),
            'num_classes': booster.num_classes()
        } 