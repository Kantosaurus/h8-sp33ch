import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LassoClassifierModel:
    """
    Lasso Classifier model for hate speech classification
    Uses L1 regularization for feature selection and sparse solutions
    """
    
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            penalty='l1',  # Lasso regularization
            solver='liblinear',
            C=1.0,  # Inverse of regularization strength
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the Lasso Classifier model with optional hyperparameter tuning
        """
        print("Training Lasso Classifier...")
        
        # Scale features for Lasso
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'max_iter': [1000, 2000]
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(
                    random_state=42,
                    penalty='l1',
                    solver='liblinear'
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
        
        print(f"Lasso Classifier CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("Lasso Classifier Results:")
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
    
    def get_selected_features(self, feature_names=None, threshold=0.0):
        """
        Get features selected by Lasso (non-zero coefficients)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting selected features")
        
        coefficients = self.model.coef_[0]
        selected_indices = np.where(np.abs(coefficients) > threshold)[0]
        
        if feature_names is not None:
            selected_features = [feature_names[i] for i in selected_indices]
            selected_coefficients = coefficients[selected_indices]
            
            selected_df = pd.DataFrame({
                'feature': selected_features,
                'coefficient': selected_coefficients,
                'abs_coefficient': np.abs(selected_coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            return selected_df
        else:
            return {
                'indices': selected_indices,
                'coefficients': coefficients[selected_indices]
            }
    
    def get_sparsity_info(self):
        """
        Get information about model sparsity
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting sparsity info")
        
        coefficients = self.model.coef_[0]
        n_features = len(coefficients)
        n_nonzero = np.count_nonzero(coefficients)
        sparsity_ratio = 1 - (n_nonzero / n_features)
        
        return {
            'total_features': n_features,
            'non_zero_features': n_nonzero,
            'zero_features': n_features - n_nonzero,
            'sparsity_ratio': sparsity_ratio
        }
    
    def get_model_params(self):
        """
        Get current model parameters
        """
        return {
            'C': self.model.C,
            'penalty': self.model.penalty,
            'solver': self.model.solver,
            'max_iter': self.model.max_iter,
            'best_params': self.best_params
        } 