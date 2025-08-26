import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class KNNModel:
    """
    K-Nearest Neighbors model for hate speech classification
    Good for capturing local patterns and non-linear decision boundaries
    """
    
    def __init__(self, random_state=42):
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2  # Euclidean distance
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the KNN model with optional hyperparameter tuning
        """
        print("Training K-Nearest Neighbors...")
        
        # Scale features for KNN (important for distance-based algorithms)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],  # Manhattan and Euclidean distance
                'leaf_size': [20, 30, 40]
            }
            
            grid_search = GridSearchCV(
                KNeighborsClassifier(),
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
        
        print(f"KNN CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("KNN Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_neighbors_info(self, X, k=5):
        """
        Get information about k nearest neighbors for given samples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting neighbors info")
        
        X_scaled = self.scaler.transform(X)
        distances, indices = self.model.kneighbors(X_scaled, n_neighbors=k)
        
        return {
            'distances': distances,
            'indices': indices
        }
    
    def get_model_params(self):
        """
        Get current model parameters
        """
        return {
            'n_neighbors': self.model.n_neighbors,
            'weights': self.model.weights,
            'algorithm': self.model.algorithm,
            'leaf_size': self.model.leaf_size,
            'p': self.model.p
        }

if __name__ == "__main__":
    from data_utils import load_and_preprocess_data
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Create and train model
    model = KNNModel()
    cv_results = model.train(X_train, y_train)
    
    # Evaluate model
    test_results = model.evaluate(X_test, y_test)
    
    print("\nKNN Model Training Complete!")