"""
PCA with Adam Optimizer - Hate Speech Classification
Model: Principal Component Analysis + Logistic Regression with Adam-like optimization
Performance: Dimensionality reduction with adaptive gradient optimization
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


class PCAAdamClassifier:
    def __init__(
        self,
        n_components=500,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_iter=2000,
    ):
        """
        PCA + Adam-optimized Logistic Regression Classifier

        Args:
            n_components: Number of SVD components to retain
            learning_rate: Learning rate for Adam optimizer
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            max_iter: Maximum iterations for SGD
        """
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="ascii",
        )

        # TruncatedSVD for dimensionality reduction (better for sparse matrices)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

        # SGD Classifier with Adam-like optimization
        # Note: sklearn doesn't have pure Adam, but we can use SGD with similar properties
        self.classifier = SGDClassifier(
            loss="log_loss",  # Logistic regression
            learning_rate="adaptive",  # Adaptive learning rate
            eta0=self.learning_rate,  # Initial learning rate
            alpha=0.0001,  # L2 regularization
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
        )

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization"""
        return self.vectorizer.transform(texts)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the PCA + Adam classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training PCA + Adam Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Apply SVD for dimensionality reduction
        print(f"Applying TruncatedSVD (reducing to {self.n_components} components)...")
        X_train_svd = self.svd.fit_transform(X_train_tfidf)

        print(
            f"Dimensionality reduced from {X_train_tfidf.shape[1]} to {X_train_svd.shape[1]} features"
        )
        print(
            f"SVD explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}"
        )

        # Train classifier
        print("Training classifier with Adam-like optimization...")
        self.classifier.fit(X_train_svd, y_train)

        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_tfidf = self.vectorizer.transform(X_val)
            X_val_svd = self.svd.transform(X_val_tfidf)
            val_predictions = self.classifier.predict(X_val_svd)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_tfidf = self.preprocess_data(X)
        X_svd = self.svd.transform(X_tfidf)
        return self.classifier.predict(X_svd)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_tfidf = self.preprocess_data(X)
        X_svd = self.svd.transform(X_tfidf)
        return self.classifier.predict_proba(X_svd)

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)
        X_svd = self.svd.fit_transform(X_tfidf)

        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier,
            X_svd,
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

    def save_model(self, model_path, vectorizer_path, svd_path):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(svd_path, "wb") as f:
            pickle.dump(self.svd, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"SVD transformer saved to {svd_path}")


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
    print("PCA + ADAM OPTIMIZER HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "pca_outputs"
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
    print(f"\nInitializing PCA + Adam Classifier...")
    print(f"Configuration:")
    print(f"  - TruncatedSVD components: 500")
    print(f"  - Learning rate: 0.01")
    print(f"  - Adam beta1: 0.9")
    print(f"  - Adam beta2: 0.999")
    print(f"  - Max iterations: 2000")
    print(f"  - Class weight: balanced")

    model = PCAAdamClassifier(
        n_components=500,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        max_iter=2000,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

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

    submission_path = os.path.join(output_dir, "submission_pca.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "pca_model.pkl")
    vectorizer_path = os.path.join(output_dir, "pca_vectorizer.pkl")
    svd_path = os.path.join(output_dir, "pca_svd.pkl")

    model.save_model(model_path, vectorizer_path, svd_path)

    # Save metrics
    metrics = {
        "Model": "pca",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "PCA_Components": model.n_components,
        "Explained_Variance": float(model.svd.explained_variance_ratio_.sum()),
        "Learning_Rate": model.learning_rate,
        "Max_Iter": model.max_iter,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("PCA + ADAM CLASSIFIER SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"TruncatedSVD Components Used: {model.n_components}")
    print(f"Explained Variance: {model.svd.explained_variance_ratio_.sum():.4f}")
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ PCA + Adam classifier training and prediction completed!")


if __name__ == "__main__":
    main()
