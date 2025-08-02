import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class BaggingClassifierModel:
    """
    Bagging Classifier model for hate speech classification
    Combines multiple base estimators with bootstrap sampling
    """
    
    def __init__(self, random_state=42):
        base_estimator = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=10,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.cv_scores = None
        self.best_params = None
        
    def train(self, X_train, y_train, cv_folds=5, tune_hyperparameters=True):
        """
        Train the Bagging Classifier model with optional hyperparameter tuning
        """
        print("Training Bagging Classifier...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [5, 10, 15, 20],
                'max_samples': [0.7, 0.8, 0.9, 1.0],
                'max_features': [0.7, 0.8, 0.9, 1.0],
                'base_estimator__max_depth': [5, 10, 15],
                'base_estimator__min_samples_split': [2, 5, 10]
            }
            
            base_estimator = DecisionTreeClassifier(random_state=42)
            
            grid_search = GridSearchCV(
                BaggingClassifier(
                    base_estimator=base_estimator,
                    bootstrap=True,
                    bootstrap_features=False,
                    random_state=42,
                    n_jobs=-1
                ),
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
        
        print(f"Bagging Classifier CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("Bagging Classifier Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (average across all estimators)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Average feature importance across all estimators
        all_importances = []
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                all_importances.append(estimator.feature_importances_)
        
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
        else:
            avg_importance = np.zeros(X_train.shape[1])
        
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return avg_importance
    
    def get_estimator_predictions(self, X):
        """
        Get predictions from individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting estimator predictions")
        
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_estimator_probabilities(self, X):
        """
        Get probability predictions from individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting estimator probabilities")
        
        probabilities = []
        for estimator in self.model.estimators_:
            if hasattr(estimator, 'predict_proba'):
                prob = estimator.predict_proba(X)[:, 1]  # Probability of positive class
                probabilities.append(prob)
        
        return np.array(probabilities)
    
    def get_model_info(self):
        """
        Get detailed model information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting model info")
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_samples': self.model.max_samples,
            'max_features': self.model.max_features,
            'bootstrap': self.model.bootstrap,
            'bootstrap_features': self.model.bootstrap_features,
            'oob_score': self.model.oob_score,
            'base_estimator_type': type(self.model.base_estimator).__name__,
            'best_params': self.best_params
        }
    
    def get_estimator_variance(self, X):
        """
        Get variance of predictions across estimators (measure of uncertainty)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting estimator variance")
        
        predictions = self.get_estimator_predictions(X)
        variance = np.var(predictions, axis=0)
        return variance 