"""
AdaBoost Classifier - Hate Speech Classification
Model: Adaptive Boosting with Decision Trees
Performance: Boosting ensemble method
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class AdaBoostHateSpeechClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=1.0,
        algorithm="SAMME.R",
        max_depth=1,  # For base estimator
    ):
        """
        AdaBoost Classifier

        Args:
            n_estimators: Number of weak learners
            learning_rate: Learning rate shrinks contribution of each classifier
            algorithm: Boosting algorithm ('SAMME', 'SAMME.R')
            max_depth: Max depth for base decision tree estimators
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.max_depth = max_depth

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Reduced for AdaBoost efficiency
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="ascii",
        )

        # Base estimator (Decision Tree)
        base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        # AdaBoost Classifier
        self.classifier = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=42,
        )

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization"""
        return self.vectorizer.transform(texts)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the AdaBoost classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training AdaBoost Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        print(
            f"Sparsity: {1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.4f}"
        )

        # Train AdaBoost
        print(f"Training AdaBoost with {self.n_estimators} estimators...")
        self.classifier.fit(X_train_tfidf, y_train)

        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_tfidf = self.preprocess_data(X_val)
            val_predictions = self.classifier.predict(X_val_tfidf)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_tfidf = self.preprocess_data(X)
        return self.classifier.predict(X_tfidf)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_tfidf = self.preprocess_data(X)
        return self.classifier.predict_proba(X_tfidf)

    def decision_function(self, X):
        """Get decision function values"""
        X_tfidf = self.preprocess_data(X)
        return self.classifier.decision_function(X_tfidf)

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)

        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier,
            X_tfidf,
            y,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
        )

        print(f"Cross-validation scores: {cv_scores}")
        print(
            f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        return cv_scores

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3):
        """Perform hyperparameter tuning"""
        print("Starting hyperparameter tuning...")

        # Transform data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Base estimator for grid search
        base_estimator = DecisionTreeClassifier(random_state=42)

        # Parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.5, 1.0, 1.5, 2.0],
            "base_estimator__max_depth": [1, 2, 3, 4],
            "algorithm": ["SAMME", "SAMME.R"],
        }

        grid_search = GridSearchCV(
            AdaBoostClassifier(base_estimator=base_estimator, random_state=42),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train_tfidf, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update classifier with best parameters
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance"""
        if not hasattr(self.classifier, "feature_importances_"):
            print("Model not trained yet!")
            return None

        importances = self.classifier.feature_importances_

        if feature_names is None:
            feature_names = self.vectorizer.get_feature_names_out()

        # Get top important features
        indices = np.argsort(importances)[::-1][:top_n]

        print(f"\nTop {top_n} most important features:")
        print("-" * 50)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx][:30]:<30} : {importances[idx]:8.4f}")

        return dict(zip([feature_names[i] for i in indices], importances[indices]))

    def get_estimator_weights(self, top_n=10):
        """Get weights of individual estimators"""
        if not hasattr(self.classifier, "estimator_weights_"):
            print("Model not trained yet!")
            return None

        weights = self.classifier.estimator_weights_

        print(f"\nEstimator weights (top {min(top_n, len(weights))}):")
        print("-" * 30)
        for i, weight in enumerate(weights[:top_n]):
            print(f"Estimator {i+1:2d}: {weight:8.4f}")

        return weights

    def get_estimator_errors(self, top_n=10):
        """Get errors of individual estimators"""
        if not hasattr(self.classifier, "estimator_errors_"):
            print("Model not trained yet!")
            return None

        errors = self.classifier.estimator_errors_

        print(f"\nEstimator errors (top {min(top_n, len(errors))}):")
        print("-" * 30)
        for i, error in enumerate(errors[:top_n]):
            print(f"Estimator {i+1:2d}: {error:8.4f}")

        return errors

    def save_model(self, model_path, vectorizer_path):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")


def load_data(train_path, test_path):
    """Load training and testing data"""
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def main():
    """Main execution function"""
    print("=" * 60)
    print("ADABOOST HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "adaboost_outputs"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Load data
    train_data, test_data = load_data(train_path, test_path)

    # Prepare training data
    X_train_full = train_data["post"].fillna("")
    y_train_full = train_data["label"]

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Initialize and train model
    print(f"\nInitializing AdaBoost Classifier...")
    print(f"Configuration:")
    print(f"  - Number of estimators: 100")
    print(f"  - Learning rate: 1.0")
    print(f"  - Algorithm: SAMME.R")
    print(f"  - Base estimator max depth: 1 (stumps)")

    model = AdaBoostHateSpeechClassifier(
        n_estimators=100,
        learning_rate=1.0,
        algorithm="SAMME.R",
        max_depth=1,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Hyperparameter tuning (optional - uncomment if needed)
    # best_params, best_score = model.hyperparameter_tuning(X_train, y_train)

    # Get model insights
    feature_importance = model.get_feature_importance(top_n=15)
    estimator_weights = model.get_estimator_weights(top_n=10)
    estimator_errors = model.get_estimator_errors(top_n=10)

    # Validation predictions
    print("\nEvaluating on validation set...")
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_predictions, average="weighted"
    )

    print(f"\nValidation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-score: {val_f1:.4f}")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, val_predictions))

    # Test predictions
    print(f"\nMaking predictions on test set...")
    X_test = test_data["post"].fillna("")
    test_predictions = model.predict(X_test)

    # Create submission file
    submission_df = pd.DataFrame({"id": test_data["id"], "label": test_predictions})

    submission_path = os.path.join(output_dir, "submission_adaboost.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "adaboost_model.pkl")
    vectorizer_path = os.path.join(output_dir, "adaboost_vectorizer.pkl")

    model.save_model(model_path, vectorizer_path)

    # Save metrics
    metrics = {
        "Model": "adaboost",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "N_Estimators": model.n_estimators,
        "Learning_Rate": model.learning_rate,
        "Algorithm": model.algorithm,
        "Base_Max_Depth": model.max_depth,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("ADABOOST CLASSIFIER SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Number of Estimators: {model.n_estimators}")
    print(f"Learning Rate: {model.learning_rate}")
    print(f"Algorithm: {model.algorithm}")
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ AdaBoost classifier training and prediction completed!")


if __name__ == "__main__":
    main()
