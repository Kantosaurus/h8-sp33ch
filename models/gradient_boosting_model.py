import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class GradientBoostingModel:
    """
    Gradient Boosting model for hate speech classification
    Excellent performance with sequential weak learners
    """
    
    def __init__(self, random_state=42):
        self.model = GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=1.0
        )
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the gradient boosting model with optional hyperparameter tuning
        """
        print("Training Gradient Boosting...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
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
        
        print(f"Gradient Boosting CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("Gradient Boosting Results:")
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
    
    def get_staged_predictions(self, X, n_stages=None):
        """
        Get predictions from each stage of the boosting process
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting staged predictions")
        
        if n_stages is None:
            n_stages = self.model.n_estimators
        
        staged_preds = []
        for pred in self.model.staged_predict(X):
            staged_preds.append(pred)
            if len(staged_preds) >= n_stages:
                break
        
        return np.array(staged_preds)
    
    def get_staged_probabilities(self, X, n_stages=None):
        """
        Get probability predictions from each stage of the boosting process
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting staged probabilities")
        
        if n_stages is None:
            n_stages = self.model.n_estimators
        
        staged_probs = []
        for prob in self.model.staged_predict_proba(X):
            staged_probs.append(prob[:, 1])  # Probability of positive class
            if len(staged_probs) >= n_stages:
                break
        
        return np.array(staged_probs)
    
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
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'subsample': self.model.subsample,
            'best_params': self.best_params
        }

if __name__ == "__main__":
    from data_utils import load_and_preprocess_data
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
    
    # Create and train model
    model = GradientBoostingModel()
    cv_results = model.train(X_train, y_train)
    
    # Evaluate model
    test_results = model.evaluate(X_test, y_test)
    
    print("\nGradient Boosting Model Training Complete!")