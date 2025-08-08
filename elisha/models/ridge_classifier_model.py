import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class RidgeClassifierModel:
    """
    Ridge Classifier model for hate speech classification
    Good for high-dimensional data with L2 regularization
    """
    
    def __init__(self, random_state=42):
        self.base_model = RidgeClassifier(
            random_state=random_state,
            alpha=1.0,
            solver='auto',
            max_iter=1000
        )
        # Calibrated classifier for probability estimates
        self.model = CalibratedClassifierCV(
            self.base_model,
            cv=3,
            method='sigmoid'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the Ridge Classifier model with optional hyperparameter tuning
        """
        print("Training Ridge Classifier...")
        
        # Scale features for Ridge Classifier
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'base_estimator__alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'base_estimator__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
            
            grid_search = GridSearchCV(
                CalibratedClassifierCV(
                    RidgeClassifier(random_state=42),
                    cv=3,
                    method='sigmoid'
                ),
                param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        
        # Perform cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=skf, scoring='f1')
        
        self.cv_scores = {
            'f1_mean': cv_scores.mean(),
            'f1_std': cv_scores.std(),
            'f1_scores': cv_scores
        }
        
        # Train on full dataset
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print(f"Ridge Classifier CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.cv_scores
    
    def predict(self, X):
        """
        Make binary predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_decision_function(self, X):
        """
        Get decision function values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
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
        
        print("Ridge Classifier Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get coefficients from the base estimator
        base_estimator = self.model.base_estimator_
        importance = np.abs(base_estimator.coef_[0])
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return importance
    
    def get_model_params(self):
        """
        Get current model parameters
        """
        base_estimator = self.model.base_estimator_
        return {
            'alpha': base_estimator.alpha,
            'solver': base_estimator.solver,
            'max_iter': base_estimator.max_iter,
            'best_params': self.best_params
        }
    
    def get_intercept(self):
        """
        Get model intercept
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting intercept")
        return self.model.base_estimator_.intercept_[0] 