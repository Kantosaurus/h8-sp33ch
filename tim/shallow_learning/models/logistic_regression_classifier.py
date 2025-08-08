"""
Logistic Regression Classifier - Hate Speech Classification
Model: Logistic Regression with PCA and TruncatedSVD dimension reduction
Performance: Linear classifier with regularization and dimensionality reduction options
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")


class LogisticRegressionHateSpeechClassifier:
    def __init__(
        self,
        dim_reduction="truncatedsvd",  # 'pca', 'truncatedsvd', 'none'
        n_components=500,
        C=1.0,
        max_iter=1000,
    ):
        """
        Logistic Regression Classifier with dimensionality reduction options

        Args:
            dim_reduction: Type of dimensionality reduction ('pca', 'truncatedsvd', 'none')
            n_components: Number of components for dimensionality reduction
            C: Regularization strength (inverse of regularization)
            max_iter: Maximum iterations for convergence
        """
        self.dim_reduction = dim_reduction
        self.n_components = n_components
        self.C = C
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

        # Dimensionality reduction
        if dim_reduction == "pca":
            self.reducer = PCA(n_components=n_components, random_state=42)
        elif dim_reduction == "truncatedsvd":
            self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            self.reducer = None

        # Logistic Regression Classifier
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",  # Handle class imbalance
            solver="liblinear",  # Good for small datasets and L1 regularization
        )

        # Store training metrics
        self.training_metrics = {}

    def preprocess_data(self, texts):
        """Apply TF-IDF vectorization and dimensionality reduction"""
        X_tfidf = self.vectorizer.transform(texts)

        if self.reducer is not None:
            if self.dim_reduction == "pca":
                X_reduced = self.reducer.transform(X_tfidf.toarray())
            else:  # TruncatedSVD
                X_reduced = self.reducer.transform(X_tfidf)
            return X_reduced
        else:
            return X_tfidf

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Logistic Regression classifier

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        print("Training Logistic Regression Classifier...")
        start_time = time.time()

        # Transform texts to TF-IDF
        print("Applying TF-IDF vectorization...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Apply dimensionality reduction if specified
        if self.reducer is not None:
            print(
                f"Applying {self.dim_reduction.upper()} (reducing to {self.n_components} components)..."
            )

            if self.dim_reduction == "pca":
                X_train_reduced = self.reducer.fit_transform(X_train_tfidf.toarray())
            else:  # TruncatedSVD
                X_train_reduced = self.reducer.fit_transform(X_train_tfidf)

            print(
                f"Dimensionality reduced from {X_train_tfidf.shape[1]} to {X_train_reduced.shape[1]} features"
            )

            if hasattr(self.reducer, "explained_variance_ratio_"):
                print(
                    f"Explained variance ratio: {self.reducer.explained_variance_ratio_.sum():.4f}"
                )
        else:
            X_train_reduced = X_train_tfidf
            print(
                f"No dimensionality reduction applied. Using {X_train_tfidf.shape[1]} features"
            )

        # Train classifier
        print("Training Logistic Regression...")
        self.classifier.fit(X_train_reduced, y_train)

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

    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")

        # Transform data
        X_tfidf = self.vectorizer.fit_transform(X)

        if self.reducer is not None:
            if self.dim_reduction == "pca":
                X_processed = self.reducer.fit_transform(X_tfidf.toarray())
            else:
                X_processed = self.reducer.fit_transform(X_tfidf)
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

        if self.reducer is not None:
            if self.dim_reduction == "pca":
                X_train_processed = self.reducer.fit_transform(X_train_tfidf.toarray())
            else:
                X_train_processed = self.reducer.fit_transform(X_train_tfidf)
        else:
            X_train_processed = X_train_tfidf

        # Parameter grid
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }

        grid_search = GridSearchCV(
            LogisticRegression(
                random_state=42, max_iter=self.max_iter, class_weight="balanced"
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

    def get_feature_importance(self, feature_names=None, top_n=20):
        """Get feature importance (coefficients)"""
        if not hasattr(self.classifier, "coef_"):
            print("Model not trained yet!")
            return None

        coef = self.classifier.coef_[0]

        if feature_names is None:
            if self.reducer is None:
                feature_names = self.vectorizer.get_feature_names_out()
            else:
                feature_names = [f"component_{i}" for i in range(len(coef))]

        # Get top positive and negative coefficients
        indices = np.argsort(np.abs(coef))[::-1][:top_n]

        print(f"\nTop {top_n} most important features:")
        print("-" * 50)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx][:30]:<30} : {coef[idx]:8.4f}")

        return dict(zip([feature_names[i] for i in indices], coef[indices]))

    def save_model(self, model_path, vectorizer_path, reducer_path=None):
        """Save the trained model components"""
        with open(model_path, "wb") as f:
            pickle.dump(self.classifier, f)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        if self.reducer is not None and reducer_path is not None:
            with open(reducer_path, "wb") as f:
                pickle.dump(self.reducer, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        if reducer_path:
            print(f"Dimensionality reducer saved to {reducer_path}")


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
    print("LOGISTIC REGRESSION HATE SPEECH CLASSIFICATION")
    print("=" * 60)

    # Set up paths
    data_dir = "../../data"
    output_dir = "logistic_regression_outputs"
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

    # Initialize and train model with TruncatedSVD (better for sparse matrices)
    print(f"\nInitializing Logistic Regression Classifier...")
    print(f"Configuration:")
    print(f"  - Dimensionality reduction: TruncatedSVD")
    print(f"  - Components: 500")
    print(f"  - Regularization (C): 1.0")
    print(f"  - Max iterations: 1000")
    print(f"  - Class weight: balanced")

    model = LogisticRegressionHateSpeechClassifier(
        dim_reduction="truncatedsvd",
        n_components=500,
        C=1.0,
        max_iter=1000,
    )

    # Train the model
    model.fit(X_train, y_train, X_val, y_val)

    # Cross-validation on full training data
    cv_scores = model.cross_validate(X_train_full, y_train_full, cv_folds=5)

    # Hyperparameter tuning (optional)
    # best_params, best_score = model.hyperparameter_tuning(X_train, y_train)

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

    # Feature importance
    feature_importance = model.get_feature_importance(top_n=15)

    # Test predictions
    print(f"\nMaking predictions on test set...")
    X_test = test_data["post"].fillna("")
    test_predictions = model.predict(X_test)

    # Create submission file
    submission_df = pd.DataFrame({"id": test_data["id"], "label": test_predictions})

    submission_path = os.path.join(output_dir, "submission_logistic_regression.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    # Save model components
    model_path = os.path.join(output_dir, "logistic_regression_model.pkl")
    vectorizer_path = os.path.join(output_dir, "logistic_regression_vectorizer.pkl")
    reducer_path = os.path.join(output_dir, "logistic_regression_reducer.pkl")

    model.save_model(model_path, vectorizer_path, reducer_path)

    # Save metrics
    metrics = {
        "Model": "logistic_regression",
        "Accuracy": float(val_accuracy),
        "Precision": float(val_precision),
        "Recall": float(val_recall),
        "F1": float(val_f1),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "Dim_Reduction": model.dim_reduction,
        "N_Components": model.n_components,
        "C": model.C,
        "Max_Iter": model.max_iter,
    }

    if hasattr(model.reducer, "explained_variance_ratio_"):
        metrics["Explained_Variance"] = float(
            model.reducer.explained_variance_ratio_.sum()
        )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")

    # Print final summary
    print(f"\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFIER SUMMARY")
    print("=" * 60)
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f}")
    print(f"Dimensionality Reduction: {model.dim_reduction}")
    if model.reducer:
        print(f"Components Used: {model.n_components}")
        if hasattr(model.reducer, "explained_variance_ratio_"):
            print(
                f"Explained Variance: {model.reducer.explained_variance_ratio_.sum():.4f}"
            )
    print(f"Test Predictions: {len(test_predictions)} samples")

    # Print prediction distribution
    unique, counts = np.unique(test_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    if len(unique) == 2:
        hate_ratio = counts[1] / len(test_predictions) if 1 in unique else 0
        print(f"Predicted hate speech ratio: {hate_ratio:.4f}")

    print(f"\n✅ Logistic Regression classifier training and prediction completed!")


if __name__ == "__main__":
    main()
