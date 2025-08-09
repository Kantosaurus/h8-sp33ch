"""
SVM Classifier - Hate Speech Classification
Model: Support Vector Machine with PCA dimensionality reduction
Performance: SVM with linear and RBF kernels and PCA preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class SVMHateSpeechClassifier:
    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        n_components=300,
        use_pca=True,
    ):
        """
        SVM Classifier with PCA dimensionality reduction

        Args:
            kernel: SVM kernel ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            n_components: Number of PCA components
            use_pca: Whether to use PCA for dimensionality reduction
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.use_pca = use_pca

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Reduced for SVM efficiency
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents="ascii",
        )

        # PCA for dimensionality reduction (essential for SVM with text data)
        if use_pca:
            self.pca = PCA(n_components=n_components, random_state=42)
        else:
            self.pca = None

        # SVM Classifier
        self.classifier = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
            probability=True,  # Enable probability predictions
        )

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization and PCA"""
        X_tfidf = self.vectorizer.transform(texts)

        if self.pca is not None:
            X_pca = self.pca.transform(X_tfidf.toarray())
            return X_pca
        else:
            return X_tfidf

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the SVM classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training SVM Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Apply PCA if specified
        if self.pca is not None:
            print(f"Applying PCA (reducing to {self.n_components} components)...")
            X_train_pca = self.pca.fit_transform(X_train_tfidf.toarray())

            print(
                f"Dimensionality reduced from {X_train_tfidf.shape[1]} to {X_train_pca.shape[1]} features"
            )
            print(
                f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}"
            )

            X_train_processed = X_train_pca
        else:
            X_train_processed = X_train_tfidf
            print(f"No PCA applied. Using {X_train_tfidf.shape[1]} features")

        # Train SVM
        print(f"Training SVM with {self.kernel} kernel...")
        self.classifier.fit(X_train_processed, y_train)

        # Validation performance
        if X_val is not None and y_val is not None:
            X_val_processed = self.preprocess_data(X_val)
            val_predictions = self.classifier.predict(X_val_processed)
            val_accuracy = accuracy_score(y_val, val_predictions)
            print(f"Validation accuracy: {val_accuracy:.4f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self

    def predict(self, X):
        """Make predictions on new data"""
        X_processed = self.preprocess_data(X)
        return self.classifier.predict(X_processed)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_processed = self.preprocess_data(X)
        return self.classifier.predict_proba(X_processed)

    def decision_function(self, X):
        """Get decision function values"""
        X_processed = self.preprocess_data(X)
        return self.classifier.decision_function(X_processed)

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)

        if self.pca is not None:
            X_processed = self.pca.fit_transform(X_tfidf.toarray())
        else:
            X_processed = X_tfidf

        # Cross-validation
        cv_scores = cross_val_score(
            self.classifier,
            X_processed,
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

        if self.pca is not None:
            X_train_processed = self.pca.fit_transform(X_train_tfidf.toarray())
        else:
            X_train_processed = X_train_tfidf

        # Parameter grid for different kernels
        if self.kernel == "rbf":
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            }
        elif self.kernel == "linear":
            param_grid = {"C": [0.1, 1.0, 10.0, 100.0]}
        else:
            param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}

        grid_search = GridSearchCV(
            SVC(
                kernel=self.kernel,
                random_state=42,
                class_weight="balanced",
                probability=True,
            ),
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train_processed, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        # Update classifier with best parameters
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def get_support_vectors_info(self):
        """Get information about support vectors"""
        if not hasattr(self.classifier, "support_"):
            print("Model not trained yet!")
            return None

        n_support = self.classifier.n_support_
        support_vectors = self.classifier.support_vectors_

        print(f"\nSupport Vector Information:")
        print(f"Number of support vectors per class: {n_support}")
        print(f"Total support vectors: {len(self.classifier.support_)}")
        print(
            f"Support vector ratio: {len(self.classifier.support_) / len(self.classifier.support_vectors_):.4f}"
        )

        return {
            "n_support": n_support,
            "total_support_vectors": len(self.classifier.support_),
            "support_vector_shape": support_vectors.shape,
        }

    def save_model(self, model_path, vectorizer_path, pca_path=None):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        if self.pca is not None and pca_path is not None:
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        if pca_path:
            print(f"PCA transformer saved to {pca_path}")


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
    print("SVM HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "svm_outputs"
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
    print(f"\nInitializing SVM Classifier...")
    print(f"Configuration:")
    print(f"  - Kernel: RBF")
    print(f"  - PCA components: 300")
    print(f"  - C (regularization): 1.0")
    print(f"  - Gamma: scale")
    print(f"  - Class weight: balanced")

    model = SVMHateSpeechClassifier(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        n_components=300,
        use_pca=True,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Hyperparameter tuning (optional - uncomment if needed)
    # best_params, best_score = model.hyperparameter_tuning(X_train, y_train)

    # Support vector information
    sv_info = model.get_support_vectors_info()

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

    submission_path = os.path.join(output_dir, "submission_svm.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "svm_model.pkl")
    vectorizer_path = os.path.join(output_dir, "svm_vectorizer.pkl")
    pca_path = os.path.join(output_dir, "svm_pca.pkl")

    model.save_model(model_path, vectorizer_path, pca_path)

    # Save metrics
    metrics = {
        "Model": "svm",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "Kernel": model.kernel,
        "C": model.C,
        "Gamma": str(model.gamma),
        "PCA_Components": model.n_components,
    }

    if model.pca is not None:
        metrics["Explained_Variance"] = float(model.pca.explained_variance_ratio_.sum())

    if sv_info:
        metrics["Support_Vectors"] = int(sv_info["total_support_vectors"])

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("SVM CLASSIFIER SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Kernel: {model.kernel}")
    if model.pca:
        print(f"PCA Components Used: {model.n_components}")
        print(
            f"PCA Explained Variance: {model.pca.explained_variance_ratio_.sum():.4f}"
        )
    if sv_info:
        print(f"Support Vectors: {sv_info['total_support_vectors']}")
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ SVM classifier training and prediction completed!")


if __name__ == "__main__":
    main()
