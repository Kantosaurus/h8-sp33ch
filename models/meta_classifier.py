import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class MetaClassifier:
    """
    Meta-classifier that learns from base model outputs
    Uses Logistic Regression to combine predictions with meta-features
    """
    
    def __init__(self, random_state=42):
        self.meta_model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.base_models = {}
        self.meta_features_names = []
        
    def add_base_model(self, name, model):
        """
        Add a base model to the ensemble
        """
        self.base_models[name] = model
        
    def extract_meta_features(self, X):
        """
        Extract meta-features from base model predictions
        """
        if not self.base_models:
            raise ValueError("No base models added to meta-classifier")
        
        meta_features = []
        predictions = []
        
        # Get predictions from all base models
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                pred_proba = model.decision_function(X)
            else:
                raise ValueError(f"Model {name} must have predict_proba or decision_function method")
            
            predictions.append(pred_proba)
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Meta-features:
        # 1. Raw probabilities from each model
        meta_features.append(predictions)
        
        # 2. Model disagreement (standard deviation of predictions)
        disagreement = np.std(predictions, axis=1, keepdims=True)
        meta_features.append(disagreement)
        
        # 3. Confidence gap (difference between max and min predictions)
        confidence_gap = np.max(predictions, axis=1, keepdims=True) - np.min(predictions, axis=1, keepdims=True)
        meta_features.append(confidence_gap)
        
        # 4. Prediction variance
        prediction_variance = np.var(predictions, axis=1, keepdims=True)
        meta_features.append(prediction_variance)
        
        # 5. Mean prediction
        mean_prediction = np.mean(predictions, axis=1, keepdims=True)
        meta_features.append(mean_prediction)
        
        # 6. Median prediction
        median_prediction = np.median(predictions, axis=1, keepdims=True)
        meta_features.append(median_prediction)
        
        # 7. Range of predictions
        prediction_range = np.max(predictions, axis=1, keepdims=True) - np.min(predictions, axis=1, keepdims=True)
        meta_features.append(prediction_range)
        
        # 8. Number of models predicting above threshold
        threshold = 0.5
        above_threshold = np.sum(predictions > threshold, axis=1, keepdims=True)
        meta_features.append(above_threshold)
        
        # Combine all meta-features
        combined_features = np.hstack(meta_features)
        
        # Store feature names for interpretability
        if not self.meta_features_names:
            self.meta_features_names = []
            # Add base model prediction names
            for name in self.base_models.keys():
                self.meta_features_names.append(f"{name}_pred")
            # Add meta-feature names
            self.meta_features_names.extend([
                'model_disagreement',
                'confidence_gap',
                'prediction_variance',
                'mean_prediction',
                'median_prediction',
                'prediction_range',
                'models_above_threshold'
            ])
        
        return combined_features
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the meta-classifier on base model outputs
        """
        print("Training Meta-Classifier...")
        
        # Extract meta-features for training
        X_meta_train = self.extract_meta_features(X_train)
        
        # Scale features
        X_meta_train_scaled = self.scaler.fit_transform(X_meta_train)
        
        # Train meta-classifier
        self.meta_model.fit(X_meta_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            X_meta_val = self.extract_meta_features(X_val)
            X_meta_val_scaled = self.scaler.transform(X_meta_val)
            
            y_pred_meta = self.meta_model.predict(X_meta_val_scaled)
            f1_meta = f1_score(y_val, y_pred_meta)
            accuracy_meta = accuracy_score(y_val, y_pred_meta)
            
            print(f"Meta-classifier validation F1: {f1_meta:.4f}")
            print(f"Meta-classifier validation Accuracy: {accuracy_meta:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using meta-classifier
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before making predictions")
        
        X_meta = self.extract_meta_features(X)
        X_meta_scaled = self.scaler.transform(X_meta)
        
        return self.meta_model.predict(X_meta_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities from meta-classifier
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before making predictions")
        
        X_meta = self.extract_meta_features(X)
        X_meta_scaled = self.scaler.transform(X_meta)
        
        return self.meta_model.predict_proba(X_meta_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate meta-classifier performance
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("Meta-Classifier Results:")
        for metric, value in results.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def get_feature_importance(self):
        """
        Get meta-feature importance
        """
        if not self.is_trained:
            raise ValueError("Meta-classifier must be trained before getting feature importance")
        
        importance = np.abs(self.meta_model.coef_[0])
        
        importance_df = pd.DataFrame({
            'feature': self.meta_features_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_base_model_predictions(self, X):
        """
        Get predictions from all base models
        """
        predictions = {}
        
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                pred_proba = model.decision_function(X)
            else:
                raise ValueError(f"Model {name} must have predict_proba or decision_function method")
            
            predictions[name] = pred_proba
        
        return predictions
    
    def analyze_model_agreement(self, X):
        """
        Analyze agreement between base models
        """
        predictions = self.get_base_model_predictions(X)
        
        # Convert to binary predictions
        binary_predictions = {}
        for name, pred in predictions.items():
            binary_predictions[name] = (pred > 0.5).astype(int)
        
        # Calculate agreement matrix
        model_names = list(binary_predictions.keys())
        n_models = len(model_names)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                agreement = np.mean(binary_predictions[model_names[i]] == binary_predictions[model_names[j]])
                agreement_matrix[i, j] = agreement
        
        return agreement_matrix, model_names 