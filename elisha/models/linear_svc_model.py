import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class LinearSVCModel:
    """
    Linear Support Vector Classifier model for hate speech classification
    Good for high-dimensional data and linear separability
    """
    
    def __init__(self, random_state=42):
        self.base_model = LinearSVC(
            random_state=random_state,
            C=1.0,
            loss='squared_hinge',
            max_iter=1000,
            dual=True
        )
        # Calibrated classifier for probability estimates
        self.model = CalibratedClassifierCV(
            self.base_model,
            cv=3,
            method='sigmoid'
        )
        self.scaler = MaxAbsScaler()
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=False):
        """
        Train the Linear SVC model with optional hyperparameter tuning
        """
        print("Training Linear SVC...")
        
        # Scale features for SVC
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'base_estimator__C': [0.1, 0.5, 1.0, 2.0, 5.0],
                'base_estimator__loss': ['hinge', 'squared_hinge'],
                'base_estimator__max_iter': [1000, 2000]
            }
            
            grid_search = GridSearchCV(
                CalibratedClassifierCV(
                    LinearSVC(random_state=42),
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
        
        print(f"Linear SVC CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        Get decision function values (distance from hyperplane)
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
        
        print("Linear SVC Results:")
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
    
    def get_support_vectors_info(self, X_train, y_train):
        """
        Get information about support vectors
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting support vectors info")
        
        X_train_scaled = self.scaler.transform(X_train)
        base_estimator = self.model.base_estimator_
        
        # Get support vector indices
        support_indices = base_estimator.support_
        support_vectors = X_train_scaled[support_indices]
        
        return {
            'n_support_vectors': len(support_indices),
            'support_indices': support_indices,
            'support_vectors': support_vectors,
            'support_labels': y_train[support_indices]
        }
    
    def get_model_params(self):
        """
        Get current model parameters
        """
        base_estimator = self.model.base_estimator_
        return {
            'C': base_estimator.C,
            'loss': base_estimator.loss,
            'max_iter': base_estimator.max_iter,
            'dual': base_estimator.dual,
            'best_params': self.best_params
        } 