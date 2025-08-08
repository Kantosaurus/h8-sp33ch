import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class VotingClassifierModel:
    """
    Voting Classifier model for hate speech classification
    Combines multiple different base models with voting mechanism
    """
    
    def __init__(self, random_state=42):
        # Define base estimators
        estimators = [
            ('lr', LogisticRegression(random_state=random_state, max_iter=1000)),
            ('dt', DecisionTreeClassifier(random_state=random_state, max_depth=10)),
            ('svc', SVC(random_state=random_state, probability=True)),
            ('nb', MultinomialNB())
        ]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            weights=None  # Equal weights for all estimators
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cv_scores = None
        
    def train(self, X_train, y_train, cv_folds=5):
        """
        Train the Voting Classifier model
        """
        print("Training Voting Classifier...")
        
        # Scale features for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        
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
        
        print(f"Voting Classifier CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        print("Voting Classifier Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_individual_predictions(self, X):
        """
        Get predictions from individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting individual predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'predict'):
                pred = estimator.predict(X_scaled)
                predictions[name] = pred
        
        return predictions
    
    def get_individual_probabilities(self, X):
        """
        Get probability predictions from individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting individual probabilities")
        
        X_scaled = self.scaler.transform(X)
        probabilities = {}
        
        for name, estimator in self.model.named_estimators_.items():
            if hasattr(estimator, 'predict_proba'):
                prob = estimator.predict_proba(X_scaled)[:, 1]  # Probability of positive class
                probabilities[name] = prob
        
        return probabilities
    
    def get_estimator_weights(self):
        """
        Get current estimator weights
        """
        return self.model.weights
    
    def set_estimator_weights(self, weights):
        """
        Set custom weights for estimators
        """
        if len(weights) != len(self.model.estimators):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of estimators ({len(self.model.estimators)})")
        
        self.model.weights = weights
    
    def get_model_info(self):
        """
        Get detailed model information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting model info")
        
        estimator_info = []
        for name, estimator in self.model.named_estimators_.items():
            estimator_info.append({
                'name': name,
                'type': type(estimator).__name__,
                'params': estimator.get_params()
            })
        
        return {
            'voting': self.model.voting,
            'weights': self.model.weights,
            'estimators': estimator_info
        }
    
    def get_consensus_analysis(self, X):
        """
        Analyze consensus among individual estimators
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting consensus analysis")
        
        individual_preds = self.get_individual_predictions(X)
        
        # Convert to array for analysis
        pred_array = np.array(list(individual_preds.values()))
        
        # Calculate consensus metrics
        consensus_ratio = np.mean(pred_array, axis=0)  # Average prediction
        agreement_count = np.sum(pred_array == pred_array[0], axis=0)  # Number of agreeing estimators
        disagreement_ratio = 1 - (agreement_count / len(individual_preds))
        
        return {
            'consensus_ratio': consensus_ratio,
            'agreement_count': agreement_count,
            'disagreement_ratio': disagreement_ratio,
            'individual_predictions': individual_preds
        } 